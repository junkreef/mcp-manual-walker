from pathlib import Path

import pytest
import requests
from fastmcp.client import Client

from mcp_manual_walker import config
from mcp_manual_walker.database import init_db
from mcp_manual_walker.main import (
    app,
    get_manual_metadata,
    get_markdown_content,
    list_manuals,
    sync_database,
)


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
    # Clear the cache directory before each test
    for item in cache_dir.iterdir():
        if item.is_file():
            item.unlink()

    # 2. Download the reference PDF
    pdf_url = "https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf"
    pdf_path = pdf_dir / "PDF32000_2008.pdf"
    response = requests.get(pdf_url)
    with open(pdf_path, "wb") as f:
        f.write(response.content)


    # 3. Configure app settings to use temporary paths
    original_pdf_dir = config.settings.PDF_ROOT_DIR
    original_db_path = config.settings.DB_FILE_PATH
    original_cache_dir = config.settings.CACHE_DIR

    config.settings.PDF_ROOT_DIR = pdf_dir
    config.settings.DB_FILE_PATH = db_dir / "test.db"
    config.settings.CACHE_DIR = cache_dir

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
    assert manual["file_name"] == "PDF32000_2008.pdf"
    assert manual["document_title"] == "PDF32000.book"
    manual_id = manual["id"]

    # 2. Get manual metadata
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    metadata = result.structured_content
    assert metadata["id"] == manual_id
    assert "table_of_contents" in metadata
    toc = metadata["table_of_contents"]
    assert len(toc) > 1

    # Find a specific bookmark to test
    # Let's find "1.1 Introduction"
    intro_bookmark = None
    for item in toc:
        if "Introduction" in item["title"]:
            intro_bookmark = item
            break

    assert intro_bookmark is not None, "Could not find 'Introduction' bookmark"
    bookmark_id = intro_bookmark["id"]


    # 3. Get markdown content for a specific bookmark
    result = await test_client.call_tool("get_markdown_content", {"bookmark_id": bookmark_id})
    content = result.structured_content['result']
    assert isinstance(content, str)
    # Since markitdown adds a title, we check if it's in the content
    assert "Introduction" in content
    assert len(content) > 100


@pytest.mark.asyncio
async def test_delete_orphaned_cache(tmp_path: Path):
    """
    Tests that orphaned cache files are deleted when the source PDF is removed.
    """
    # 1. Create a temporary environment
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    cache_dir = tmp_path / "cache"
    pdf_dir.mkdir()
    db_dir.mkdir()
    cache_dir.mkdir()

    # 2. Download the reference PDF
    pdf_url = "https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf"
    pdf_path = pdf_dir / "PDF32000_2008.pdf"
    response = requests.get(pdf_url)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # 3. Configure settings
    original_pdf_dir = config.settings.PDF_ROOT_DIR
    original_db_path = config.settings.DB_FILE_PATH
    original_cache_dir = config.settings.CACHE_DIR
    config.settings.PDF_ROOT_DIR = pdf_dir
    config.settings.DB_FILE_PATH = db_dir / "test.db"
    config.settings.CACHE_DIR = cache_dir

    # 4. Initialize app and database
    init_db()
    sync_database()

    # 5. Get manual and bookmark IDs
    manuals = list_manuals.fn()
    manual_id = manuals[0]['id']
    metadata = get_manual_metadata.fn(manual_id)
    toc = metadata['table_of_contents']
    scope_bookmark = None
    for item in toc:
        if "Scope" in item["title"]:
            scope_bookmark = item
            break
    assert scope_bookmark is not None, "Could not find 'Scope' bookmark"
    bookmark_id = scope_bookmark['id']

    # 6. Create a cache file
    get_markdown_content.fn(bookmark_id)
    cache_files = list(cache_dir.glob("*.md"))
    assert len(cache_files) == 1
    cache_file_path = cache_files[0]
    assert cache_file_path.exists()

    # 7. Delete the source PDF
    pdf_path.unlink()

    # 8. Run sync_database again to trigger the cleanup
    sync_database()

    # 9. Verify the cache file has been deleted
    assert not cache_file_path.exists()

    # Restore original settings
    config.settings.PDF_ROOT_DIR = original_pdf_dir
    config.settings.DB_FILE_PATH = original_db_path
    config.settings.CACHE_DIR = original_cache_dir