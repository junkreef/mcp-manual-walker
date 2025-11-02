import shutil
from pathlib import Path

import pytest
from fastmcp.client import Client

from mcp_manual_walker import config
from mcp_manual_walker.database import SessionLocal
from mcp_manual_walker.main import app
from mcp_manual_walker.sync import sync_database


@pytest.fixture(scope="function")
async def test_client(tmp_path: Path, monkeypatch, dummy_pdf_factory):
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

    # 2. Create a dummy PDF with enough pages for pagination testing
    pdf_path = pdf_dir / "dummy_manual.pdf"
    pages_content = {i: f"Content for page {i}" for i in range(1, 31)}
    dummy_pdf_factory(
        path=pdf_path,
        pages_content=pages_content,
        bookmarks={
            "Chapter 1": (1, None),      # Spans pages 1-14
            "Section 1.1": (2, "Chapter 1"),
            "Chapter 2": (15, None),     # Spans pages 15-24
            "Section 2.1": (16, "Chapter 2"),
            "Chapter 3": (25, None),     # Spans pages 25-30
        },
    )

    # 3. Configure app settings to use temporary paths
    monkeypatch.setattr(config.settings, "PDF_ROOT_DIR", pdf_dir)
    monkeypatch.setattr(config.settings, "DB_FILE_PATH", db_dir / "test.db")
    monkeypatch.setattr(config.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(config.settings, "MAX_PAGES_PER_REQUEST", 5) # For predictable tests

    # 4. Manually run sync_database to populate the test DB
    sync_database()

    # 5. Yield an in-memory client
    async with Client(app) as client:
        yield client


@pytest.mark.asyncio
async def test_e2e_workflow(test_client: Client):
    """
    Tests the basic end-to-end workflow with the new paginated response.
    """
    # 1. List manuals
    result = await test_client.call_tool("list_manuals")
    manuals = result.structured_content['result']
    assert isinstance(manuals, list)
    assert len(manuals) == 1
    manual = manuals[0]
    assert manual["file_name"] == "dummy_manual.pdf"
    manual_id = manual["id"]

    # 2. Get manual metadata
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    metadata = result.structured_content
    assert metadata["id"] == manual_id
    toc = metadata["table_of_contents"]
    bookmark_id = toc[0]["children"][0]["id"]  # Section 1.1

    # 3. Get markdown content (first page)
    result = await test_client.call_tool("get_markdown_content", {"bookmark_id": bookmark_id})
    content_response = result.structured_content
    assert isinstance(content_response, dict)
    assert "markdown_content" in content_response
    assert "Content for page 2" in content_response["markdown_content"]


@pytest.mark.asyncio
async def test_pagination_workflow(test_client: Client):
    """
    Tests the new pagination feature thoroughly.
    """
    # 1. Get the bookmark ID for Chapter 1, which spans 14 pages.
    result = await test_client.call_tool("list_manuals")
    manual_id = result.structured_content['result'][0]["id"]
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    chapter1_bookmark_id = result.structured_content["table_of_contents"][0]["id"]

    # 2. First request: should return the first 5 pages (due to MAX_PAGES_PER_REQUEST)
    result = await test_client.call_tool("get_markdown_content", {"bookmark_id": chapter1_bookmark_id})
    res1 = result.structured_content
    assert res1["bookmark_total_pages"] == 14
    assert res1["page_offset"] == 0
    assert res1["page_limit"] == 5
    assert res1["next_page_offset"] == 5
    assert "Content for page 1" in res1["markdown_content"]
    assert "Content for page 5" in res1["markdown_content"]
    assert "Content for page 6" not in res1["markdown_content"]

    # 3. Second request: use offset to get the next chunk
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_offset": 5}
    )
    res2 = result.structured_content
    assert res2["page_offset"] == 5
    assert res2["page_limit"] == 5
    assert res2["next_page_offset"] == 10
    assert "Content for page 6" in res2["markdown_content"]
    assert "Content for page 10" in res2["markdown_content"]
    assert "Content for page 11" not in res2["markdown_content"]

    # 4. Third request: get the last few pages
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_offset": 10}
    )
    res3 = result.structured_content
    assert res3["page_offset"] == 10
    assert res3["page_limit"] == 4 # Only 4 pages left
    assert res3["next_page_offset"] is None # This is the last chunk
    assert "Content for page 11" in res3["markdown_content"]
    assert "Content for page 14" in res3["markdown_content"]

    # 5. Request with a specific, smaller page_limit
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_limit": 2}
    )
    res4 = result.structured_content
    assert res4["page_limit"] == 2
    assert res4["next_page_offset"] == 2

    # 6. Out of bounds request
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_offset": 99}
    )
    res5 = result.structured_content
    assert "error" in res5
    assert "out of bounds" in res5["error"]


@pytest.mark.asyncio
async def test_delete_orphaned_cache(test_client: Client):
    """
    Tests that the orphaned cache directory is deleted when the source PDF is removed.
    """
    # The test_client fixture has already run sync_database once.
    # 1. Get IDs and create a cache entry to ensure the cache directory exists.
    result = await test_client.call_tool("list_manuals")
    manual_id = result.structured_content['result'][0]["id"]
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    bookmark_id = result.structured_content["table_of_contents"][0]["id"]

    await test_client.call_tool("get_markdown_content", {"bookmark_id": bookmark_id})

    # 2. Verify the cache directory now exists
    manual_cache_dir = config.settings.CACHE_DIR / manual_id
    assert manual_cache_dir.exists()
    assert manual_cache_dir.is_dir()
    assert len(list(manual_cache_dir.iterdir())) > 0

    # 3. Delete the source PDF
    pdf_path = config.settings.PDF_ROOT_DIR / "dummy_manual.pdf"
    pdf_path.unlink()

    # 4. Run sync_database again to trigger the cleanup
    sync_database()

    # 5. Verify the cache directory has been deleted
    assert not manual_cache_dir.exists()