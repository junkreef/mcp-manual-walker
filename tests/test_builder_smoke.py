import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from mcp_manual_walker import database
from mcp_manual_walker.config import settings


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
    ):
        yield settings

    # Cleanup: dispose engine created by builder.init_db() via database module
    if database.engine:
        database.engine.dispose()


def test_builder_smoke(tmp_path, mock_settings):
    """Smoke test for builder.py using MOCKED dependencies."""

    # 1. Prepare Paths
    # 1. Prepare Paths
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    sub_dir = pdf_dir / "subdir"
    sub_dir.mkdir(exist_ok=True)

    # Create dummy PDFs
    (pdf_dir / "sample.pdf").write_text("dummy")
    (sub_dir / "nested.pdf").write_text("dummy")

    # 2. Prepare Mocks
    mock_docling = MagicMock()
    mock_docling_conv = MagicMock()
    mock_docling.document_converter = mock_docling_conv

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

    # 3. Patch sys.modules and Run
    with patch.dict(
        sys.modules,
        {
            "docling": mock_docling,
            "docling.document_converter": mock_docling_conv,
            "langchain_text_splitters": mock_lts,
            "chromadb": mock_chromadb,
            "chromadb.utils": mock_chromadb.utils,
            "chromadb.utils.embedding_functions": mock_chromadb.utils.embedding_functions,
        },
    ):
        # Import/Reload builder inside patched environment
        from mcp_manual_walker import builder

        importlib.reload(builder)

        # Patch local imports inside builder (which are now 'real' imports in the module object)
        with (
            patch("mcp_manual_walker.builder.extract_pdf_metadata") as mock_meta,
            patch("mcp_manual_walker.builder.calculate_file_hash") as mock_hash,
            patch("mcp_manual_walker.builder.chunk_text_by_coordinates") as mock_chunk,
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

            # Use the mocked dependencies via builder attribute if they were imported directly
            # builder.DocumentConverter is from docling.document_converter
            # Since we patched sys.modules BEFORE reload, builder should have picked up the mocks.
            # builder.py:
            # try: from docling.document_converter import DocumentConverter ...

            # Since mock_docling_conv is in sys.modules, import should succeed and grab the mock.

            # Run build twice to check coverage? Or just once with True
            builder.build(pdf_dir, reset=True, save_markdown=True)

            # Verify
            assert mock_inst.convert.call_count == 2
            assert mock_coll.add.call_count == 2

            # Check args of last call
            call_args = mock_coll.add.call_args
            kwargs = call_args[1]

            assert len(kwargs["ids"]) == 2
            metas = kwargs["metadatas"]
            # Check metadata keys
            assert "bookmark_id" in metas[0]
            assert metas[0]["bookmark_id"] == "bm1"
            assert metas[1]["bookmark_id"] == "bm2"

            # Check manual_id
            assert "manual_id" in metas[0]

