from pathlib import Path

import pytest
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

    # 2. Create a dummy PDF for testing
    pdf_path = pdf_dir / "dummy_manual.pdf"
    dummy_pdf_factory(
        path=pdf_path,
        pages_content={
            1: "Content for page 1",
            2: "Content for page 2",
            3: "Content for page 3",
            4: "Content for page 4",
            5: "Content for page 5",
        },
        bookmarks={
            "Chapter 1": (1, None),
            "Section 1.1": (2, "Chapter 1"),
            "Chapter 2": (3, None),
            "Section 2.1": (4, "Chapter 2"),
            "Section 2.2": (5, "Chapter 2"),
        },
    )

    # 3. Configure app settings to use temporary paths
    monkeypatch.setattr(config.settings, "PDF_ROOT_DIR", pdf_dir)
    monkeypatch.setattr(config.settings, "DB_FILE_PATH", db_dir / "test.db")
    monkeypatch.setattr(config.settings, "CACHE_DIR", cache_dir)

    # 4. The app's lifespan manager will handle init_db and sync_database.
    # 5. Yield an in-memory client
    async with Client(app) as client:
        yield client


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
    assert manual["file_name"] == "dummy_manual.pdf"
    manual_id = manual["id"]

    # 2. Get manual metadata
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    metadata = result.structured_content
    assert metadata["id"] == manual_id
    assert "table_of_contents" in metadata
    toc = metadata["table_of_contents"]
    assert len(toc) == 2  # Chapter 1 and Chapter 2

    # Find a specific bookmark to test
    chapter1 = toc[0]
    assert chapter1["title"] == "Chapter 1"
    assert len(chapter1["children"]) == 1
    section1_1 = chapter1["children"][0]
    assert section1_1["title"] == "Section 1.1"
    bookmark_id = section1_1["id"]

    # 3. Get markdown content for a specific bookmark
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": bookmark_id}
    )
    content = result.structured_content['result']
    assert isinstance(content, str)
    assert content  # Check that the content is not empty


@pytest.mark.asyncio
async def test_delete_orphaned_cache(tmp_path: Path, monkeypatch, dummy_pdf_factory):
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

    # 2. Create a dummy PDF
    pdf_path = pdf_dir / "dummy_manual.pdf"
    dummy_pdf_factory(
        path=pdf_path,
        pages_content={1: "Page 1"},
        bookmarks={"Chapter 1": (1, None)}
    )

    # 3. Configure settings
    monkeypatch.setattr(config.settings, "PDF_ROOT_DIR", pdf_dir)
    monkeypatch.setattr(config.settings, "DB_FILE_PATH", db_dir / "test.db")
    monkeypatch.setattr(config.settings, "CACHE_DIR", cache_dir)

    # 4. Initialize app and database
    init_db()
    sync_database()

    # 5. Get manual and bookmark IDs
    manuals = list_manuals.fn()
    manual_id = manuals[0]['id']
    metadata = get_manual_metadata.fn(manual_id)
    toc = metadata['table_of_contents']
    bookmark_id = toc[0]['id']

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