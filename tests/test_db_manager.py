import json
from argparse import Namespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mcp_manual_walker.config import settings
from mcp_manual_walker.db_manager import (
    command_delete,
    command_export,
    command_import,
    command_list,
)
from mcp_manual_walker.models import Manual


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


def test_command_export(mock_session, mock_chroma):
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

    mock_session.execute.return_value.scalars.return_value.all.return_value = [manual]

    # Mock Chroma get
    collection.get.return_value = {
        "ids": ["c1", "c2"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "metadatas": [{"m": 1}, {"m": 2}],
        "documents": ["doc1", "doc2"],
    }

    args = Namespace(target="exp.pdf", output="out.zip")

    # Mock zipfile and open
    with (
        patch("zipfile.ZipFile") as mock_zip_cls,
        patch("builtins.open", mock_open()),
        patch("mcp_manual_walker.db_manager.Path"),
        patch("tempfile.TemporaryDirectory") as mock_temp,
    ):
        # Setup mocks
        mock_zip_instance = mock_zip_cls.return_value.__enter__.return_value
        mock_temp_path = MagicMock()
        mock_temp.return_value.__enter__.return_value = mock_temp_path

        # Act
        command_export(args)

        # Verify
        assert mock_zip_instance.write.call_count == 3


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
    chroma_data = {
        "ids": ["c1"],
        "embeddings": [[0.1]],
        "metadatas": [{"manual_id": "uuid-imp"}],
        "documents": ["chunk"],
    }

    args = Namespace(input="in.zip")

    with (
        patch("zipfile.ZipFile"),
        patch("tempfile.TemporaryDirectory"),
        patch("builtins.open", mock_open()),
        patch("json.load") as mock_json_load,
        patch("mcp_manual_walker.db_manager.Path") as MockPath,
        patch("mcp_manual_walker.db_manager.get_embedder"),
    ):
        # Setup mocks
        mock_path_inst = MockPath.return_value
        mock_path_inst.exists.return_value = True

        mock_json_load.side_effect = [manifest_data, sqlite_data, chroma_data]

        # FIX: Ensure query returns None so it doesn't skip
        mock_session.scalars.return_value.first.return_value = None

        # Act
        command_import(args)

        # Assert
        assert mock_session.add.call_count >= 1

        collection.add.assert_called()
        call_args = collection.add.call_args
        assert call_args.kwargs["ids"] == ["c1"]


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
    chroma_data = {
        "ids": ["c1"],
        "embeddings": [[0.1]],
        "metadatas": [{"manual_id": "uuid-imp"}],
        "documents": ["chunk"],
    }

    args = Namespace(input="in.zip")

    with (
        patch("zipfile.ZipFile"),
        patch("tempfile.TemporaryDirectory"),
        patch("builtins.open", mock_open()),
        patch("json.load") as mock_json_load,
        patch("mcp_manual_walker.db_manager.Path") as MockPath,
        patch("mcp_manual_walker.db_manager.get_embedder") as mock_get_embedder,
    ):
        mock_path_inst = MockPath.return_value
        mock_path_inst.exists.return_value = True

        mock_json_load.side_effect = [manifest_data, sqlite_data, chroma_data]
        mock_session.scalars.return_value.first.return_value = None

        command_import(args)

        # Nothing is written: neither the relational rows nor the vectors.
        collection.add.assert_not_called()
        mock_session.add.assert_not_called()
        mock_get_embedder.assert_not_called()
