import concurrent.futures
import importlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mcp_manual_walker import database
from mcp_manual_walker.config import settings
from mcp_manual_walker.models import Figure, Manual

# Size of the image the fake Docling picture renders.
FAKE_FIGURE_SIZE = (8, 6)


class FakeBBox:
    """Stand-in for a docling bbox (PDF points, bottom-left origin)."""

    def __init__(self, left, bottom, right, top):
        self.l = left  # noqa: E741 - mirrors the docling attribute name
        self.b = bottom
        self.r = right
        self.t = top


class FakePicture:
    """Minimal stand-in for docling's PictureItem."""

    def __init__(self, index=0, page=2):
        self.self_ref = f"#/pictures/{index}"
        self.prov = [
            SimpleNamespace(page_no=page, bbox=FakeBBox(10.0, 20.0, 110.0, 120.0))
        ]
        # An ImageRef on the real item; only its removal matters here.
        self.image = object()

    def get_image(self, doc):
        return Image.new("RGB", FAKE_FIGURE_SIZE, (200, 30, 30))


def _fake_save_as_markdown(path, artifacts_dir=None, image_mode=None):
    """Writes a markdown dump the way DoclingDocument.save_as_markdown does."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("MD Content", encoding="utf-8")
    if artifacts_dir is not None:
        # Docling resolves a relative artifacts_dir against the markdown file.
        (path.parent / artifacts_dir).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def mock_settings(tmp_path):
    # Patch attributes of the existing settings object so that
    # all modules (including database.py which presumably has already imported it)
    # see the updated values.

    new_db_path = tmp_path / "test.db"
    new_pdf_dir = tmp_path / "pdfs"
    new_chroma_path = tmp_path / "chroma_db"

    new_markdown_path = tmp_path / "markdown"

    # Use patch.object on the instance
    with (
        patch.object(settings, "DB_FILE_PATH", new_db_path),
        patch.object(settings, "PDF_ROOT_DIR", new_pdf_dir),
        patch.object(settings, "CHROMADB_PATH", new_chroma_path),
        patch.object(settings, "MARKDOWN_OUTPUT_DIR", new_markdown_path),
        # Keep the pipeline single-worker and in-process for deterministic tests
        patch.object(settings, "METADATA_WORKERS", 1),
        patch.object(settings, "DOCLING_WORKERS", 1),
    ):
        yield settings

    # Cleanup: dispose engine created by builder.init_db() via database module
    if database.engine:
        database.engine.dispose()


def _thread_executor(max_workers, initializer=None, initargs=()):
    """
    Stand-in for builder._make_process_executor.

    Spawned processes cannot see the mocks installed in this process, so the
    tests run the whole pipeline on threads instead.
    """
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=initargs,
    )


@pytest.fixture
def builder_env(mock_settings):
    """
    Reloads builder.py with docling/langchain/chromadb replaced by mocks and
    yields a namespace with the handles the tests need to assert on.
    """
    mock_docling = MagicMock()
    mock_docling_conv = MagicMock()
    mock_docling.document_converter = mock_docling_conv

    mock_docling_dm = MagicMock()
    mock_docling.datamodel = mock_docling_dm
    mock_docling.datamodel.base_models = MagicMock()
    mock_docling.datamodel.pipeline_options = MagicMock()

    mock_docling_backend = MagicMock()
    mock_docling.backend = mock_docling_backend
    mock_docling_backend.docling_parse_backend = MagicMock()

    mock_lts = MagicMock()

    mock_chromadb = MagicMock()
    mock_chromadb.utils = MagicMock()
    mock_chromadb.utils.embedding_functions = MagicMock()

    # Mock Docling behavior
    mock_inst = MagicMock()
    mock_res = MagicMock()
    mock_res.document.export_to_markdown.return_value = "MD Content"
    # One rendered picture, so every conversion returns a figure record.
    mock_res.document.pictures = [FakePicture(index=0, page=2)]
    mock_res.document.save_as_markdown.side_effect = _fake_save_as_markdown
    mock_inst.convert.return_value = mock_res
    mock_docling_conv.DocumentConverter.return_value = mock_inst

    # Mock ChromaDB behavior
    mock_client_inst = MagicMock()
    mock_coll = MagicMock()
    mock_client_inst.get_or_create_collection.return_value = mock_coll
    mock_chromadb.PersistentClient.return_value = mock_client_inst

    pipeline_options = mock_docling.datamodel.pipeline_options

    with patch.dict(
        sys.modules,
        {
            "docling": mock_docling,
            "docling.document_converter": mock_docling_conv,
            "docling.datamodel": mock_docling_dm,
            "docling.datamodel.base_models": mock_docling.datamodel.base_models,
            "docling.datamodel.pipeline_options": pipeline_options,
            "docling.backend": mock_docling_backend,
            "docling.backend.docling_parse_backend": (
                mock_docling_backend.docling_parse_backend
            ),
            "langchain_text_splitters": mock_lts,
            "chromadb": mock_chromadb,
            "chromadb.utils": mock_chromadb.utils,
            "chromadb.utils.embedding_functions": (
                mock_chromadb.utils.embedding_functions
            ),
        },
    ):
        # Import/Reload builder inside patched environment so that the guarded
        # module level imports pick up the mocks.
        from mcp_manual_walker import builder

        importlib.reload(builder)

        # Stand-in for the sentence-transformers embedder: no model download.
        fake_embedder = SimpleNamespace(
            embed_documents=lambda docs: [[0.1, 0.2, 0.3] for _ in docs],
            model_name=settings.EMBEDDING_MODEL,
            dimension=3,
        )
        # get_or_create_collection returns the metadata of an already existing
        # collection, so the mock has to look like a matching one.
        mock_coll.metadata = {"embedding_model": settings.EMBEDDING_MODEL}

        with (
            patch("mcp_manual_walker.pdf_utils.extract_pdf_metadata") as mock_meta,
            patch("mcp_manual_walker.pdf_utils.calculate_file_hash") as mock_hash,
            patch("mcp_manual_walker.builder.chunk_document") as mock_chunk,
            patch(
                "mcp_manual_walker.builder._make_process_executor",
                side_effect=_thread_executor,
            ),
            patch(
                "mcp_manual_walker.builder.get_embedder",
                return_value=fake_embedder,
            ),
        ):
            mock_hash.return_value = "dummy_hash"
            mock_meta.return_value = {
                "document_title": "Test Doc",
                "page_count": 5,
                "bookmarks": [
                    {"title": "Intro", "level": 1, "page_num": 1, "top": 800.0}
                ],
            }
            mock_chunk.return_value = [
                {
                    "text": "Chunk 1",
                    "metadata": {
                        "manual_id": "id",
                        "bookmark_id": "bm1",
                        "type": "text",
                    },
                },
                {
                    "text": "Chunk 2",
                    "metadata": {
                        "manual_id": "id",
                        "bookmark_id": "bm2",
                        "type": "figure",
                        "page": 2,
                        "picture_index": 0,
                        "figure_caption": "Figure 1: Panel",
                        "figure_labels": "Power, Reset",
                        "figure_description": "",
                    },
                },
            ]

            yield types.SimpleNamespace(
                builder=builder,
                mock_inst=mock_inst,
                mock_client=mock_client_inst,
                mock_res=mock_res,
                mock_coll=mock_coll,
                mock_hash=mock_hash,
                mock_meta=mock_meta,
                mock_chunk=mock_chunk,
            )


@pytest.fixture
def pdf_dir(tmp_path):
    """Creates two dummy PDFs, one of them in a nested directory."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    sub_dir = pdf_dir / "subdir"
    sub_dir.mkdir(exist_ok=True)

    (pdf_dir / "sample.pdf").write_text("dummy")
    (sub_dir / "nested.pdf").write_text("dummy")
    return pdf_dir


def test_builder_smoke(pdf_dir, mock_settings, builder_env):
    """Smoke test for builder.py using MOCKED dependencies."""
    builder_env.builder.build(pdf_dir, reset=True, save_markdown=True)

    assert builder_env.mock_inst.convert.call_count == 2
    assert builder_env.mock_coll.add.call_count == 2

    # Chroma must never embed anything itself: vectors are always passed in,
    # and the collection records which model produced them.
    create_kwargs = builder_env.mock_client.get_or_create_collection.call_args.kwargs
    assert create_kwargs["embedding_function"] is None
    assert create_kwargs["metadata"]["embedding_model"] == settings.EMBEDDING_MODEL

    # Check args of last call
    kwargs = builder_env.mock_coll.add.call_args[1]

    assert len(kwargs["ids"]) == 2
    assert len(kwargs["documents"]) == 2
    metas = kwargs["metadatas"]
    assert metas[0]["bookmark_id"] == "bm1"
    assert metas[1]["bookmark_id"] == "bm2"
    assert "manual_id" in metas[0]

    # Chunk kind and figure location travel into the Chroma metadata
    assert metas[0]["type"] == "text"
    assert "page" not in metas[0]
    assert metas[1]["type"] == "figure"
    assert metas[1]["page"] == 2

    # The image itself stays in SQLite: Chroma only learns the figure id, not
    # the picture index or the caption/labels/description copies.
    assert "figure_id" in metas[1]
    for key in ("picture_index", "figure_caption", "figure_labels",
                "figure_description"):
        assert key not in metas[1]
    assert "figure_id" not in metas[0]

    # Embeddings are computed in the main process and passed explicitly
    assert len(kwargs["embeddings"]) == len(kwargs["ids"])

    # Markdown files keep the nested layout of the source tree
    md_root = mock_settings.MARKDOWN_OUTPUT_DIR
    assert (md_root / "sample.md").exists()
    assert (md_root / "subdir" / "nested.md").exists()

    # Relational DB holds both manuals with their bookmarks and figures
    with database.SessionLocal() as session:
        manuals = session.query(Manual).all()
        assert len(manuals) == 2
        for manual in manuals:
            assert len(manual.bookmarks) == 1
            assert len(manual.figures) == 1

        # The figure id in the Chroma metadata resolves to a stored PNG.
        figure = session.get(Figure, metas[1]["figure_id"])
        assert figure is not None
        assert figure.page == 2
        assert figure.picture_index == 0
        assert figure.bookmark_id == "bm2"
        assert (figure.bbox_l, figure.bbox_b, figure.bbox_r, figure.bbox_t) == (
            10.0,
            20.0,
            110.0,
            120.0,
        )
        assert figure.caption == "Figure 1: Panel"
        assert figure.labels == "Power, Reset"
        assert figure.description == ""
        assert figure.mime_type == "image/png"
        assert (figure.width, figure.height) == FAKE_FIGURE_SIZE
        stored = Image.open(io.BytesIO(figure.image))
        assert stored.format == "PNG"
        assert stored.size == FAKE_FIGURE_SIZE


def test_extract_figures_returns_png_and_strips_images(builder_env):
    """The worker helper returns PNG bytes and empties the picture images."""
    first = FakePicture(index=0, page=3)
    second = FakePicture(index=1, page=7)
    doc = SimpleNamespace(pictures=[first, second])

    figures = builder_env.builder._extract_figures(doc)

    assert len(figures) == 2
    # The index is the position in doc.pictures, so the parent can match a
    # chunk's picture_index against these records.
    assert [f["picture_index"] for f in figures] == [0, 1]
    assert [f["page"] for f in figures] == [3, 7]

    record = figures[1]
    assert record["bbox"] == (10.0, 20.0, 110.0, 120.0)
    assert (record["width"], record["height"]) == FAKE_FIGURE_SIZE

    decoded = Image.open(io.BytesIO(record["png"]))
    assert decoded.format == "PNG"
    assert decoded.size == FAKE_FIGURE_SIZE

    # The images must not travel back to the parent inside the pickled document.
    assert first.image is None
    assert second.image is None


def test_extract_figures_skips_pictures_without_image_or_prov(builder_env):
    """Unrenderable pictures are dropped, but still lose their image ref."""
    no_image = FakePicture(index=0)
    no_image.get_image = lambda doc: None
    no_prov = FakePicture(index=1)
    no_prov.prov = []
    good = FakePicture(index=2, page=4)
    doc = SimpleNamespace(pictures=[no_image, no_prov, good])

    figures = builder_env.builder._extract_figures(doc)

    # Skipped pictures do not shift the index of the ones that follow.
    assert [f["picture_index"] for f in figures] == [2]
    assert no_image.image is None
    assert no_prov.image is None


def test_builder_skips_unchanged(pdf_dir, mock_settings, builder_env):
    """A second build with identical hashes must not re-convert anything."""
    builder_env.builder.build(pdf_dir, reset=True, save_markdown=False)
    first_count = builder_env.mock_inst.convert.call_count
    assert first_count == 2

    builder_env.builder.build(pdf_dir, reset=False, save_markdown=False)

    assert builder_env.mock_inst.convert.call_count == first_count
    builder_env.mock_coll.delete.assert_not_called()


def test_builder_rebuilds_changed_manual(pdf_dir, mock_settings, builder_env):
    """A changed hash must drop the stale chunks and re-convert the file."""
    builder_env.builder.build(pdf_dir, reset=True, save_markdown=False)

    with database.SessionLocal() as session:
        manual_ids = [m.id for m in session.query(Manual).all()]
    assert manual_ids

    builder_env.mock_hash.return_value = "changed_hash"
    builder_env.builder.build(pdf_dir, reset=False, save_markdown=False)

    assert builder_env.mock_inst.convert.call_count == 4

    delete_calls = builder_env.mock_coll.delete.call_args_list
    assert len(delete_calls) == 2
    deleted_ids = {call.kwargs["where"]["manual_id"] for call in delete_calls}
    assert set(manual_ids) == deleted_ids


def test_builder_continues_after_conversion_failure(
    pdf_dir, mock_settings, builder_env
):
    """One failing conversion must not abort the whole build."""

    def convert_side_effect(path_str):
        if "nested" in path_str:
            raise RuntimeError("Simulated docling failure")
        return builder_env.mock_res

    builder_env.mock_inst.convert.side_effect = convert_side_effect

    builder_env.builder.build(pdf_dir, reset=True, save_markdown=False)

    assert builder_env.mock_inst.convert.call_count == 2
    assert builder_env.mock_coll.add.call_count == 1
