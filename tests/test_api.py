from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from mcp_manual_walker import config
from mcp_manual_walker.main import app
from mcp_manual_walker.schemas import (
    ManualInfo,
    ManualMetadata,
    MarkdownContent,
    SearchResult,
)
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
            "Chapter 1": (1, None),  # Spans pages 1-14
            "Section 1.1": (2, "Chapter 1"),
            "Chapter 2": (15, None),  # Spans pages 15-24
            "Section 2.1": (16, "Chapter 2"),
            "Chapter 3": (25, None),  # Spans pages 25-30
        },
    )

    # 3. Configure app settings to use temporary paths
    monkeypatch.setattr(config.settings, "PDF_ROOT_DIR", pdf_dir)
    monkeypatch.setattr(config.settings, "DB_FILE_PATH", db_dir / "test.db")
    monkeypatch.setattr(config.settings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        config.settings, "MAX_PAGES_PER_REQUEST", 5
    )  # For predictable tests

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
    assert result.structured_content is not None
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    assert isinstance(manuals, list)
    assert len(manuals) == 1
    manual = manuals[0]
    assert manual.file_name == "dummy_manual.pdf"
    manual_id = manual.id

    # 2. Get manual metadata
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    assert result.structured_content is not None
    metadata = ManualMetadata.model_validate(result.structured_content)
    assert metadata.id == manual_id
    toc = metadata.table_of_contents
    bookmark_id = toc[0].children[0].id  # Section 1.1

    # 3. Get markdown content (first page)
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": bookmark_id}
    )
    assert result.structured_content is not None
    content_response = MarkdownContent.model_validate(result.structured_content)
    assert content_response is not None
    assert "markdown_content" in content_response.model_dump()
    assert "Content for page 2" in content_response.markdown_content


@pytest.mark.asyncio
async def test_pagination_workflow(test_client: Client):
    """
    Tests the new pagination feature thoroughly.
    """
    # 1. Get the bookmark ID for Chapter 1, which spans 14 pages.
    result = await test_client.call_tool("list_manuals")
    assert result.structured_content is not None
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    manual_id = manuals[0].id
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    assert result.structured_content is not None
    chapter1_bookmark_id = (
        ManualMetadata.model_validate(result.structured_content).table_of_contents[0].id
    )

    # 2. First request: should return the first 5 pages (due to MAX_PAGES_PER_REQUEST)
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id}
    )
    assert result.structured_content is not None
    res1 = MarkdownContent.model_validate(result.structured_content)
    assert res1.bookmark_total_pages == 14
    assert res1.page_offset == 0
    assert res1.page_limit == 5
    assert res1.next_page_offset == 5
    assert "Content for page 1" in res1.markdown_content
    assert "Content for page 5" in res1.markdown_content
    assert "Content for page 6" not in res1.markdown_content

    # 3. Second request: use offset to get the next chunk
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_offset": 5}
    )
    assert result.structured_content is not None
    res2 = MarkdownContent.model_validate(result.structured_content)
    assert res2.page_offset == 5
    assert res2.page_limit == 5
    assert res2.next_page_offset == 10
    assert "Content for page 6" in res2.markdown_content
    assert "Content for page 10" in res2.markdown_content
    assert "Content for page 11" not in res2.markdown_content

    # 4. Third request: get the last few pages
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_offset": 10}
    )
    assert result.structured_content is not None
    res3 = MarkdownContent.model_validate(result.structured_content)
    assert res3.page_offset == 10
    assert res3.page_limit == 4  # Only 4 pages left
    assert res3.next_page_offset is None  # This is the last chunk
    assert "Content for page 11" in res3.markdown_content
    assert "Content for page 14" in res3.markdown_content

    # 5. Request with a specific, smaller page_limit
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1_bookmark_id, "page_limit": 2}
    )
    assert result.structured_content is not None
    res4 = MarkdownContent.model_validate(result.structured_content)
    assert res4.page_limit == 2
    assert res4.next_page_offset == 2

    # 6. Out of bounds request
    with pytest.raises(ToolError, match="page_offset is out of bounds."):
        await test_client.call_tool(
            "get_markdown_content",
            {"bookmark_id": chapter1_bookmark_id, "page_offset": 99},
        )


@pytest.mark.asyncio
async def test_delete_orphaned_cache(test_client: Client):
    """
    Tests that the orphaned cache directory is deleted when the source PDF is removed.
    """
    # The test_client fixture has already run sync_database once.
    # 1. Get IDs and create a cache entry to ensure the cache directory exists.
    result = await test_client.call_tool("list_manuals")
    assert result.structured_content is not None
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    manual_id = manuals[0].id
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    assert result.structured_content is not None
    bookmark_id = (
        ManualMetadata.model_validate(result.structured_content).table_of_contents[0].id
    )

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


@pytest.mark.asyncio
async def test_search_manual(test_client: Client):
    """
    Tests the search_manual tool.
    """
    # 1. Get manual ID
    result = await test_client.call_tool("list_manuals")
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    manual_id = manuals[0].id
    
    # 2. Search for "Chapter 1" (which is in the content of page 1-14 in dummy_manual)
    # Note: The dummy_pdf_factory in test_client fixture creates content like "Content for page X".
    # It does NOT put the bookmark titles in the text content automatically.
    # So we should search for "Content for page" or specific page numbers.
    
    # Search for "Content for page 2" - should be on page 2.
    # Page 2 is under "Section 1.1" (starts on page 2).
    # Note: "Content for page 2" will also match "Content for page 20", etc.
    result = await test_client.call_tool(
        "search_manual", {"manual_id": manual_id, "query": "Content for page 2"}
    )
    assert result.structured_content is not None
    search_result = SearchResult.model_validate(result.structured_content)
    
    # Find the match for page 2
    match = next((m for m in search_result.results if m.page_num == 2), None)
    assert match is not None, "Could not find match on page 2"
    assert "Content for page 2" in match.context
    
    # Verify hierarchy
    # Page 2 is the start of "Section 1.1", which is a child of "Chapter 1"
    assert len(match.bookmarks) == 2
    assert match.bookmarks[0].title == "Chapter 1"
    assert match.bookmarks[1].title == "Section 1.1"
    
    # Verify page_offset
    # Section 1.1 starts on page 2. Match is on page 2. Offset should be 0.
    assert match.page_offset == 0
    
    # 3. Search for "Content for page 3" - should be on page 3.
    # Page 3 is still under "Section 1.1" (which starts on page 2).
    result = await test_client.call_tool(
        "search_manual", {"manual_id": manual_id, "query": "Content for page 3"}
    )
    search_result = SearchResult.model_validate(result.structured_content)
    match = next((m for m in search_result.results if m.page_num == 3), None)
    assert match is not None, "Could not find match on page 3"
    
    # Verify hierarchy
    assert len(match.bookmarks) == 2
    assert match.bookmarks[1].title == "Section 1.1"
    
    # Verify page_offset
    # Section 1.1 starts on page 2. Match is on page 3. Offset should be 1.
    assert match.page_offset == 1
    
    # 4. Search for something that doesn't exist
    result = await test_client.call_tool(
        "search_manual", {"manual_id": manual_id, "query": "NonExistent"}
    )
    search_result = SearchResult.model_validate(result.structured_content)
    assert len(search_result.results) == 0


@pytest.mark.asyncio
async def test_search_manual_with_bookmark_filter(test_client: Client):
    """
    Tests the search_manual tool with bookmark filtering.
    """
    # 1. Get manual ID and bookmarks
    result = await test_client.call_tool("list_manuals")
    manual_id = ManualInfo(**result.structured_content["result"][0]).id
    
    result = await test_client.call_tool("get_manual_metadata", {"manual_id": manual_id})
    toc = ManualMetadata.model_validate(result.structured_content).table_of_contents
    
    # "Chapter 1" (page 1-14)
    chapter1_id = toc[0].id
    
    # "Section 1.1" (page 2-14, child of Chapter 1)
    section1_1_id = toc[0].children[0].id
    
    # 2. Search for "Content for page 1" restricted to Chapter 1
    # Should be found (page 1 is in Chapter 1)
    # Note: "Content for page 1" matches "Content for page 1", "Content for page 10", etc.
    # Chapter 1 spans pages 1-14.
    # Page 1 is in Chapter 1.
    # Page 10, 11, 12, 13, 14 are also in Chapter 1.
    # So we expect multiple matches.
    # Let's verify that page 1 is among them.
    result = await test_client.call_tool(
        "search_manual", 
        {"manual_id": manual_id, "query": "Content for page 1", "bookmark_id": chapter1_id}
    )
    search_result = SearchResult.model_validate(result.structured_content)
    assert len(search_result.results) > 0
    match_page_1 = next((m for m in search_result.results if m.page_num == 1), None)
    assert match_page_1 is not None
    assert match_page_1.page_num == 1
    
    # 3. Search for "Content for page 1" restricted to Section 1.1
    # Should NOT be found (Section 1.1 starts on page 2)
    # But wait, "Content for page 10" is on page 10, which IS in Section 1.1 (page 2-14).
    # So "Content for page 1" WILL match "Content for page 10" in Section 1.1.
    # We need a query that ONLY appears on page 1.
    # The dummy content is "Content for page X".
    # "Content for page 1 " (with space) might work if there is a space after number.
    # Or search for "Content for page 1." (if there is a dot, but dummy factory doesn't add dot).
    # Let's search for "Content for page 1" and verify that NO result has page_num == 1.
    result = await test_client.call_tool(
        "search_manual", 
        {"manual_id": manual_id, "query": "Content for page 1", "bookmark_id": section1_1_id}
    )
    search_result = SearchResult.model_validate(result.structured_content)
    # We expect matches for page 10, 11, etc., but NOT page 1.
    match_page_1 = next((m for m in search_result.results if m.page_num == 1), None)
    assert match_page_1 is None
    
    # 4. Search for "Content for page 2" restricted to Section 1.1
    # Should be found
    result = await test_client.call_tool(
        "search_manual", 
        {"manual_id": manual_id, "query": "Content for page 2", "bookmark_id": section1_1_id}
    )
    search_result = SearchResult.model_validate(result.structured_content)
    assert len(search_result.results) == 1
    assert search_result.results[0].page_num == 2
    
    # 5. Test invalid bookmark ID
    with pytest.raises(ToolError, match="Bookmark with id 'invalid_id' not found"):
        await test_client.call_tool(
            "search_manual", 
            {"manual_id": manual_id, "query": "test", "bookmark_id": "invalid_id"}
        )


