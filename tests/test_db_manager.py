import io
import json
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mcp_manual_walker.config import settings
from mcp_manual_walker.db_manager import (
    CHUNKS_ZST_FILE_NAME,
    EXPORT_FORMAT_VERSION,
    _chunk_source,
    command_delete,
    command_export,
    command_import,
    command_list,
)
from mcp_manual_walker.models import Base, Figure, Manual


@pytest.fixture
def mock_session():
    with patch("mcp_manual_walker.db_manager.SessionLocal") as mock:
        session_instance = mock.return_value
        yield session_instance


@pytest.fixture
def mock_chroma():
    with patch("mcp_manual_walker.db_manager.get_chroma_client") as mock:
        client_instance = mock.return_value
        collection_instance = MagicMock()
        client_instance.get_collection.return_value = collection_instance
        client_instance.get_or_create_collection.return_value = collection_instance
        yield client_instance, collection_instance


def make_session(db_path):
    """A session on a real, empty SQLite file (figures need real BLOB storage)."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_command_list_text(mock_session, capsys):
    # Setup
    manual1 = Manual(
        id="uuid1",
        file_name="doc1.pdf",
        relative_path="path/to/doc1.pdf",
        page_count=10,
        file_hash="hash1",
        updated_at=None,
    )
    manual2 = Manual(
        id="uuid2",
        file_name="doc2.pdf",
        relative_path="path/to/doc2.pdf",
        page_count=20,
        file_hash="hash2",
        updated_at=None,
    )

    mock_session.execute.return_value.scalars.return_value.all.return_value = [
        manual1,
        manual2,
    ]

    # Act
    args = Namespace(json=False)
    command_list(args)

    # Assert
    captured = capsys.readouterr()
    assert "doc1.pdf" in captured.out
    assert "doc2.pdf" in captured.out
    assert "path/to/doc1.pdf" in captured.out


def test_command_list_json(mock_session, capsys):
    # Setup
    manual1 = Manual(
        id="uuid1",
        file_name="doc1.pdf",
        relative_path="path/to/doc1.pdf",
        page_count=10,
        file_hash="hash1",
        updated_at=None,
    )

    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual1]

    # Act
    args_json = Namespace(json=True)
    command_list(args_json)

    # Assert
    captured_json = capsys.readouterr()
    output_data = json.loads(captured_json.out)
    assert len(output_data) == 1
    assert output_data[0]["file_name"] == "doc1.pdf"


def test_command_delete(mock_session, mock_chroma):
    client, collection = mock_chroma

    # Setup
    manual = Manual(
        id="uuid-del",
        file_name="del.pdf",
        relative_path="del.pdf",
        file_hash="h",
        page_count=1,
    )
    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual]

    # Act
    args = Namespace(target="del.pdf")
    command_delete(args)

    # Assert
    # Check SQLite delete
    mock_session.delete.assert_called_with(manual)
    mock_session.commit.assert_called_once()

    # Check Chroma delete
    collection.delete.assert_called_with(where={"manual_id": "uuid-del"})


def test_command_export(tmp_path, mock_session, mock_chroma):
    client, collection = mock_chroma

    # Setup Data
    manual = Manual(
        id="uuid-exp",
        file_name="exp.pdf",
        relative_path="exp.pdf",
        file_hash="h",
        page_count=5,
        updated_at=None,
        document_title="Title",
    )
    manual.bookmarks = []
    manual.figures = []

    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual]

    # Mock Chroma get
    collection.get.return_value = {
        "ids": ["c1", "c2"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "metadatas": [{"m": 1}, {"m": 2}],
        "documents": ["doc1", "doc2"],
    }

    archive = tmp_path / "out.zip"
    command_export(Namespace(target="exp.pdf", output=str(archive)))

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
        info = zf.getinfo(CHUNKS_ZST_FILE_NAME)
        with _chunk_source(zf) as source:
            chunks = list(source)

    assert sorted(names) == ["chunks.jsonl.zst", "manifest.json", "sqlite.json"]
    assert manifest["format_version"] == EXPORT_FORMAT_VERSION
    assert manifest["chunk_count"] == 2
    # The member holds a zstd frame, so the zip must not compress it again.
    assert info.compress_type == zipfile.ZIP_STORED
    assert [c["id"] for c in chunks] == ["c1", "c2"]
    assert [c["embedding"] for c in chunks] == [[0.1, 0.2], [0.3, 0.4]]


def test_command_import(mock_session, mock_chroma):
    client, collection = mock_chroma

    # Mock data
    manifest_data = {
        "version": "1.0",
        "created_at": "2024-01-01",
        "target": "t",
        "embedding_model": settings.EMBEDDING_MODEL,
    }
    sqlite_data = [
        {
            "id": "uuid-imp",
            "file_name": "imp.pdf",
            "document_title": "Title",
            "relative_path": "imp.pdf",
            "file_hash": "hash",
            "page_count": 10,
            "updated_at": "2024-01-01T00:00:00",
            "bookmarks": [],
        }
    ]
    # A real archive on disk rather than a stack of patched json.load calls:
    # the importer now chooses its path by which member the archive holds, so
    # mocking the reads out hid which layout was being exercised.
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "in.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest_data))
            zf.writestr("sqlite.json", json.dumps(sqlite_data))
            zf.writestr(
                "chunks.jsonl",
                json.dumps(
                    {
                        "id": "c1",
                        "embedding": [0.1],
                        "metadata": {"manual_id": "uuid-imp"},
                        "document": "chunk",
                    }
                ),
            )

        with patch("mcp_manual_walker.db_manager.get_embedder"):
            mock_session.scalars.return_value.first.return_value = None
            command_import(Namespace(input=str(archive)))

    assert mock_session.add.call_count >= 1
    collection.add.assert_called()
    assert collection.add.call_args.kwargs["ids"] == ["c1"]
    assert collection.add.call_args.kwargs["documents"] == ["chunk"]


def test_command_import_rejects_other_embedding_model(mock_session, mock_chroma):
    """An export built with another model must not be merged into the collection."""
    client, collection = mock_chroma

    manifest_data = {
        "version": "1.0",
        "created_at": "2024-01-01",
        "target": "t",
        "embedding_model": "intfloat/multilingual-e5-small",
    }
    sqlite_data = [
        {
            "id": "uuid-imp",
            "file_name": "imp.pdf",
            "document_title": "Title",
            "relative_path": "imp.pdf",
            "file_hash": "hash",
            "page_count": 10,
            "updated_at": "2024-01-01T00:00:00",
            "bookmarks": [],
        }
    ]
    # A real archive: the importer picks its path from what the zip holds.
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "in.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest_data))
            zf.writestr("sqlite.json", json.dumps(sqlite_data))
            zf.writestr(
                "chunks.jsonl",
                json.dumps(
                    {
                        "id": "c1",
                        "embedding": [0.1],
                        "metadata": {"manual_id": "uuid-imp"},
                        "document": "chunk",
                    }
                ),
            )

        with patch("mcp_manual_walker.db_manager.get_embedder") as mock_get_embedder:
            mock_session.scalars.return_value.first.return_value = None
            # Refused by returning, not by raising: the CLI logs and stops.
            command_import(Namespace(input=str(archive)))

    collection.add.assert_not_called()
    mock_get_embedder.assert_not_called()

def test_export_import_round_trip_with_figures(tmp_path, mock_chroma):
    """A manual with a figure survives export and import byte for byte."""
    client, collection = mock_chroma
    image = png_bytes()

    source = make_session(tmp_path / "source.db")
    manual = Manual(
        id="uuid-fig",
        file_name="fig.pdf",
        document_title="Figure manual",
        relative_path="fig.pdf",
        file_hash="hash",
        page_count=3,
    )
    manual.figures.append(
        Figure(
            id="figure-1",
            manual_id="uuid-fig",
            bookmark_id="bookmark-1",
            picture_index=2,
            page=4,
            bbox_l=10.0,
            bbox_b=20.0,
            bbox_r=110.0,
            bbox_t=120.0,
            caption="Figure 1: Panel",
            labels="Power, Reset",
            description=None,
            mime_type="image/png",
            width=8,
            height=6,
            image=image,
        )
    )
    source.add(manual)
    source.commit()

    collection.get.return_value = {
        "ids": ["uuid-fig_0"],
        "embeddings": [[0.1, 0.2]],
        "metadatas": [
            {
                "manual_id": "uuid-fig",
                "type": "figure",
                "page": 4,
                "figure_id": "figure-1",
            }
        ],
        "documents": ["Figure 1: Panel"],
    }

    archive = tmp_path / "export.zip"
    with patch("mcp_manual_walker.db_manager.SessionLocal", return_value=source):
        command_export(Namespace(target="fig.pdf", output=str(archive)))

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "figures/figure-1.png" in names
        assert zf.read("figures/figure-1.png") == image
        manifest = json.loads(zf.read("manifest.json"))
        exported = json.loads(zf.read("sqlite.json"))

    assert manifest["format_version"] == 4
    # The chunks travel one JSON object per line inside a zstd frame, not as
    # one JSON object: see EXPORT_FORMAT_VERSION. A corpus-sized archive
    # cannot be built any other way on this host.
    assert "chunks.jsonl.zst" in names
    assert manifest["figure_count"] == 1
    assert exported[0]["figures"][0]["image_file"] == "figures/figure-1.png"
    # The bytes travel as a zip member, never inside the JSON.
    assert "image" not in exported[0]["figures"][0]

    target = make_session(tmp_path / "target.db")
    with (
        patch("mcp_manual_walker.db_manager.SessionLocal", return_value=target),
        patch("mcp_manual_walker.db_manager.get_embedder"),
    ):
        command_import(Namespace(input=str(archive)))

    restored = target.get(Figure, "figure-1")
    assert restored is not None
    assert restored.image == image
    assert restored.manual_id == "uuid-fig"
    assert restored.bookmark_id == "bookmark-1"
    assert restored.picture_index == 2
    assert restored.page == 4
    assert (
        restored.bbox_l,
        restored.bbox_b,
        restored.bbox_r,
        restored.bbox_t,
    ) == (10.0, 20.0, 110.0, 120.0)
    assert restored.caption == "Figure 1: Panel"
    assert restored.labels == "Power, Reset"
    assert restored.description is None
    assert restored.mime_type == "image/png"
    assert (restored.width, restored.height) == (8, 6)

    # The chunk metadata still points at the same figure id.
    added = collection.add.call_args.kwargs
    assert added["metadatas"][0]["figure_id"] == "figure-1"

    # Deleting the manual takes its figures with it.
    target.delete(target.get(Manual, "uuid-fig"))
    target.commit()
    assert target.get(Figure, "figure-1") is None


def test_command_import_accepts_archive_without_figures(tmp_path, mock_chroma):
    """A format_version 1 archive (no figures at all) still imports."""
    client, collection = mock_chroma

    archive = tmp_path / "legacy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": "1.0",
                    "created_at": "2024-01-01T00:00:00",
                    "target": "legacy.pdf",
                    "embedding_model": settings.EMBEDDING_MODEL,
                }
            ),
        )
        zf.writestr(
            "sqlite.json",
            json.dumps(
                [
                    {
                        "id": "uuid-legacy",
                        "file_name": "legacy.pdf",
                        "document_title": "Legacy",
                        "relative_path": "legacy.pdf",
                        "file_hash": "hash",
                        "page_count": 2,
                        "updated_at": "2024-01-01T00:00:00",
                        "bookmarks": [],
                    }
                ]
            ),
        )
        zf.writestr(
            "chroma.json",
            json.dumps(
                {
                    "ids": ["uuid-legacy_0"],
                    "embeddings": [[0.1]],
                    "metadatas": [{"manual_id": "uuid-legacy"}],
                    "documents": ["chunk"],
                }
            ),
        )

    target = make_session(tmp_path / "legacy.db")
    with (
        patch("mcp_manual_walker.db_manager.SessionLocal", return_value=target),
        patch("mcp_manual_walker.db_manager.get_embedder"),
    ):
        command_import(Namespace(input=str(archive)))

    manual = target.get(Manual, "uuid-legacy")
    assert manual is not None
    assert manual.figures == []
    assert collection.add.call_args.kwargs["ids"] == ["uuid-legacy_0"]


def test_figure_images_are_stored_not_deflated(tmp_path, mock_session, mock_chroma):
    """PNG is already compressed; deflating it again is pure cost.

    Measured on the zOS/V3R1 export: 2,140 MB of PNGs deflated to 2,000 MB, a
    7% gain for roughly a third of the export's runtime.
    """
    client, collection = mock_chroma
    manual = Manual(
        id="uuid-store",
        file_name="s.pdf",
        relative_path="s.pdf",
        file_hash="h",
        page_count=1,
        updated_at=None,
        document_title="T",
    )
    manual.bookmarks = []
    figure = Figure(
        id="fig-store",
        manual_id="uuid-store",
        picture_index=0,
        page=1,
        bbox_l=0.0,
        bbox_b=0.0,
        bbox_r=1.0,
        bbox_t=1.0,
        mime_type="image/png",
        image=b"\x89PNG\r\n\x1a\n" + b"not really a png but incompressible" * 50,
    )
    manual.figures = [figure]
    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual]
    collection.get.return_value = {
        "ids": [],
        "embeddings": [],
        "metadatas": [],
        "documents": [],
    }

    archive = tmp_path / "stored.zip"
    command_export(Namespace(target="s.pdf", output=str(archive)))

    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("figures/fig-store.png")
        assert info.compress_type == zipfile.ZIP_STORED
        assert zf.read("figures/fig-store.png") == figure.image


def test_a_version_2_archive_still_imports(tmp_path, mock_session, mock_chroma):
    """chroma.json predates chunks.jsonl; those archives must keep working."""
    client, collection = mock_chroma
    archive = tmp_path / "v2.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": "1.0",
                    "format_version": 2,
                    "created_at": "2024-01-01",
                    "target": "t",
                    "embedding_model": settings.EMBEDDING_MODEL,
                }
            ),
        )
        zf.writestr(
            "sqlite.json",
            json.dumps(
                [
                    {
                        "id": "uuid-v2",
                        "file_name": "v2.pdf",
                        "document_title": "T",
                        "relative_path": "v2.pdf",
                        "file_hash": "h",
                        "page_count": 1,
                        "updated_at": None,
                        "bookmarks": [],
                    }
                ]
            ),
        )
        zf.writestr(
            "chroma.json",
            json.dumps(
                {
                    "ids": ["c-v2"],
                    "embeddings": [[0.5]],
                    "metadatas": [{"manual_id": "uuid-v2"}],
                    "documents": ["legacy chunk"],
                }
            ),
        )

    with patch("mcp_manual_walker.db_manager.get_embedder"):
        mock_session.scalars.return_value.first.return_value = None
        command_import(Namespace(input=str(archive)))

    assert collection.add.call_args.kwargs["ids"] == ["c-v2"]
    assert collection.add.call_args.kwargs["documents"] == ["legacy chunk"]


def test_import_leaves_no_temporary_file_beside_the_archive(
    tmp_path, mock_session, mock_chroma
):
    """Version 3 unpacked a 10 GB chunks.jsonl next to the archive first."""
    client, collection = mock_chroma
    archive = tmp_path / "clean.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": "1.0",
                    "format_version": 3,
                    "created_at": "2024-01-01",
                    "target": "t",
                    "embedding_model": settings.EMBEDDING_MODEL,
                }
            ),
        )
        zf.writestr("sqlite.json", json.dumps([]))
        zf.writestr("chunks.jsonl", "")

    with patch("mcp_manual_walker.db_manager.get_embedder"):
        mock_session.scalars.return_value.first.return_value = None
        command_import(Namespace(input=str(archive)))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.zip"]


def test_a_long_chunk_stream_survives_the_round_trip(
    tmp_path, mock_session, mock_chroma
):
    """More chunks than fit one zstd block, to exercise the streaming path."""
    client, collection = mock_chroma
    manual = Manual(
        id="uuid-long",
        file_name="l.pdf",
        relative_path="l.pdf",
        file_hash="h",
        page_count=1,
        updated_at=None,
        document_title="T",
    )
    manual.bookmarks = []
    manual.figures = []
    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual]

    count = 5000
    collection.get.return_value = {
        "ids": [f"c{i}" for i in range(count)],
        "embeddings": [[float(i), float(i) + 0.5] for i in range(count)],
        "metadatas": [{"manual_id": "uuid-long"} for _ in range(count)],
        "documents": [f"document body {i} " + "filler " * 20 for i in range(count)],
    }

    archive = tmp_path / "long.zip"
    command_export(Namespace(target="l.pdf", output=str(archive)))

    with zipfile.ZipFile(archive) as zf:
        with _chunk_source(zf) as source:
            chunks = list(source)

    assert len(chunks) == count
    assert chunks[0]["id"] == "c0"
    assert chunks[-1]["id"] == f"c{count - 1}"
    assert chunks[-1]["embedding"] == [float(count - 1), float(count - 1) + 0.5]
    assert chunks[2500]["document"].startswith("document body 2500 ")
