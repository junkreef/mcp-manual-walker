import concurrent.futures
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mcp_manual_walker import database
from mcp_manual_walker.config import settings
from mcp_manual_walker.models import Manual


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
            patch("mcp_manual_walker.builder.chunk_text_by_coordinates") as mock_chunk,
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
                    "metadata": {"manual_id": "id", "bookmark_id": "bm1"},
                },
                {
                    "text": "Chunk 2",
                    "metadata": {"manual_id": "id", "bookmark_id": "bm2"},
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

    # Embeddings are computed in the main process and passed explicitly
    assert len(kwargs["embeddings"]) == len(kwargs["ids"])

    # Markdown files keep the nested layout of the source tree
    md_root = mock_settings.MARKDOWN_OUTPUT_DIR
    assert (md_root / "sample.md").exists()
    assert (md_root / "subdir" / "nested.md").exists()

    # Relational DB holds both manuals with their bookmarks
    with database.SessionLocal() as session:
        manuals = session.query(Manual).all()
        assert len(manuals) == 2
        for manual in manuals:
            assert len(manual.bookmarks) == 1


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
