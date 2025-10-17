import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_in_process
from pypdf import PdfWriter

from src.mcp_manual_walker.config import settings
from src.mcp_manual_walker.main import app


def run_test_server(host: str, port: int, transport: str) -> None:
    """Function to run in the subprocess."""
    app.run(host=host, port=port, transport=transport)


@pytest.fixture
def test_env():
    """Create a temporary environment for integration tests."""
    test_data_dir = Path("./test_data")
    pdf_dir = test_data_dir / "pdfs"
    cache_dir = test_data_dir / "cache"
    db_path = test_data_dir / "test.db"

    # Create directories
    pdf_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy PDF
    pdf_path = pdf_dir / "dummy_manual.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=210, height=297)
    writer.add_blank_page(width=210, height=297)
    writer.add_metadata({"/Title": "Integration Test Manual"})
    parent_bookmark = writer.add_outline_item("Chapter 1", 0)
    writer.add_outline_item("Section 1.1", 1, parent=parent_bookmark)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Override settings
    original_settings = settings.copy()
    settings.PDF_ROOT_DIR = pdf_dir
    settings.CACHE_DIR = cache_dir
    settings.DB_FILE_PATH = db_path

    yield

    # Teardown
    shutil.rmtree(test_data_dir)
    # Restore original settings
    settings.PDF_ROOT_DIR = original_settings.PDF_ROOT_DIR
    settings.CACHE_DIR = original_settings.CACHE_DIR
    settings.DB_FILE_PATH = original_settings.DB_FILE_PATH


@pytest.mark.anyio
async def test_e2e_workflow(test_env):
    """
    Tests the full end-to-end workflow over HTTP.
    """
    with run_server_in_process(run_test_server, transport="http") as url:
        async with Client(
            transport=StreamableHttpTransport(f"{url}/mcp")
        ) as client:
            # 1. List manuals
            manuals = await client.list_manuals()
            assert len(manuals) == 1
            manual = manuals[0]
            assert manual["document_title"] == "Integration Test Manual"
            manual_id = manual["id"]

            # 2. Get manual metadata
            metadata = await client.get_manual_metadata(manual_id=manual_id)
            assert metadata["id"] == manual_id
            assert len(metadata["table_of_contents"]) == 1
            toc = metadata["table_of_contents"][0]
            assert toc["title"] == "Chapter 1"
            assert len(toc["children"]) == 1
            bookmark_id = toc["children"][0]["id"]

            # 3. Get markdown content
            content = await client.get_markdown_content(bookmark_id=bookmark_id)
            assert isinstance(content, str)
            assert content is not None

            # 4. Verify cache was created
            assert len(list(settings.CACHE_DIR.iterdir())) == 1