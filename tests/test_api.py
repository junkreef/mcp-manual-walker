import asyncio
from pathlib import Path

import pytest
from fastmcp.client import Client
from pypdf import PdfWriter


@pytest.fixture(scope="function")
async def test_client(tmp_path: Path):
    """
    A comprehensive fixture for API integration testing. It sets up a temporary
    environment with a dummy PDF, a test database, and a cache directory.
    It then initializes the database and returns an in-memory client.
    """
    # 1. Create temporary directories
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    cache_dir = tmp_path / "cache"
    pdf_dir.mkdir()
    db_dir.mkdir()
    cache_dir.mkdir()

    # 2. Create a dummy PDF with known content
    dummy_pdf_path = pdf_dir / "integration_test_manual.pdf"
    writer = PdfWriter()
    # Add a page with some text
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.drawString(10, 100, "This is the content for 1.1 Intro.")
    can.save()
    packet.seek(0)
    from pypdf import PdfReader
    new_pdf = PdfReader(packet)
    writer.add_page(new_pdf.pages[0])

    writer.add_metadata({"/Title": "Integration Test Manual"})
    bm_ch1 = writer.add_outline_item("Chapter 1", page_number=0)
    writer.add_outline_item("1.1 Intro", page_number=0, parent=bm_ch1)
    with open(dummy_pdf_path, "wb") as f:
        writer.write(f)

    # 3. Configure app settings to use temporary paths
    from mcp_manual_walker import config
    original_pdf_dir = config.settings.PDF_ROOT_DIR
    original_db_path = config.settings.DB_FILE_PATH
    original_cache_dir = config.settings.CACHE_DIR

    config.settings.PDF_ROOT_DIR = pdf_dir
    config.settings.DB_FILE_PATH = db_dir / "test.db"
    config.settings.CACHE_DIR = cache_dir

    # Import app and sync_database after patching settings
    from mcp_manual_walker.main import app, sync_database

    try:
        # 4. The app's lifespan manager will handle init_db and sync_database.
        #    We just need to yield the client.
        # 5. Yield an in-memory client
        async with Client(app) as client:
            yield client

    finally:
        # Restore original settings
        config.settings.PDF_ROOT_DIR = original_pdf_dir
        config.settings.DB_FILE_PATH = original_db_path
        config.settings.CACHE_DIR = original_cache_dir


@pytest.mark.asyncio
async def test_e2e_workflow(test_client: Client):
    """
    Tests the full end-to-end workflow:
    1. list_manuals()
    2. get_manual_metadata()
    3. get_markdown_content()
    """
    # 1. List manuals
    result = await test_client.call_tool("list_manuals")
    manuals = result.structured_content['result']
    assert isinstance(manuals, list)
    assert len(manuals) == 1
    manual = manuals[0]
    assert manual["file_name"] == "integration_test_manual.pdf"
    assert manual["document_title"] == "Integration Test Manual"
    manual_id = manual["id"]

    # 2. Get manual metadata
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    metadata = result.structured_content
    assert metadata["id"] == manual_id
    assert "table_of_contents" in metadata
    toc = metadata["table_of_contents"]
    assert len(toc) == 1
    assert toc[0]["title"] == "Chapter 1"
    assert len(toc[0]["children"]) == 1
    assert toc[0]["children"][0]["title"] == "1.1 Intro"
    bookmark_id = toc[0]["children"][0]["id"]

    # 3. Get markdown content for a specific bookmark
    result = await test_client.call_tool("get_markdown_content", {"bookmark_id": bookmark_id})
    content = result.structured_content['result']
    assert isinstance(content, str)
    # Since markitdown adds a title, we check if it's in the content
    assert "1.1 Intro" in content
    # The dummy PDF is blank, so we don't expect much other content
    assert len(content) > 10