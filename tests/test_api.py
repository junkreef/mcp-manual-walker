import uuid
from pathlib import Path

import chromadb
import pytest
from fastmcp.client import Client

from mcp_manual_walker import config, database
from mcp_manual_walker.main import app, get_embedding_function
from mcp_manual_walker.models import Bookmark, Manual
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
    environment with a dummy PDF, a test database, a ChromaDB instance, and a cache directory.
    It then initializes the database and returns an in-memory client.
    """
    # 1. Create temporary directories
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    chroma_dir = tmp_path / "chroma_db"
    pdf_dir.mkdir()
    db_dir.mkdir()
    chroma_dir.mkdir()

    # 2. Create a dummy PDF
    pdf_path = pdf_dir / "dummy_manual.pdf"
    pages_content = {
        i: f"Content for page {i}. This is unique text for search."
        for i in range(1, 31)
    }
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
    monkeypatch.setattr(config.settings, "CHROMADB_PATH", chroma_dir)
    monkeypatch.setattr(config.settings, "MAX_PAGES_PER_REQUEST", 5)

    # 4. Manually run sync_database to populate the test SQLite DB
    sync_database()

    # 5. Populate ChromaDB
    chroma_client = None
    db = database.SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.file_name == "dummy_manual.pdf").first()
        bookmarks = db.query(Bookmark).filter(Bookmark.manual_id == manual.id).all()

        # Init ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        embedding_fn = get_embedding_function()
        collection = chroma_client.get_or_create_collection(
            name="manual_chunks", embedding_function=embedding_fn
        )

        ids = []
        documents = []
        metadatas = []

        # Map page range to bookmark
        # Simple logic: iterate pages, find strict bookmark
        # For test purpose, we just need *some* chunks associated with bookmarks.

        # Process pages 1-30
        for page_num in range(1, 31):
            # Find deepest bookmark for this page
            current_bookmark = None
            for b in bookmarks:
                if b.page_num <= page_num:
                    if (
                        current_bookmark is None
                        or b.level > current_bookmark.level
                        or (
                            b.level == current_bookmark.level
                            and b.page_num > current_bookmark.page_num
                        )
                    ):
                        pass

            # Explicit mapping based on dummy structure
            b_id = None
            if 1 <= page_num <= 14:
                # Chapter 1
                if page_num == 1:
                    target_title = "Chapter 1"
                else:
                    target_title = "Section 1.1"
            elif 15 <= page_num <= 24:
                # Chapter 2
                if page_num == 15:
                    target_title = "Chapter 2"
                else:
                    target_title = "Section 2.1"
            else:
                target_title = "Chapter 3"

            found_b = next((b for b in bookmarks if b.title == target_title), None)
            b_id = found_b.id if found_b else None

            chunk_id = str(uuid.uuid4())
            content = pages_content[page_num]

            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(
                {
                    "manual_id": manual.id,
                    "bookmark_id": b_id if b_id else "",
                    "page_num": page_num,
                    "chunk_index": 0,
                }
            )

        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    finally:
        db.close()

    # 6. Yield an in-memory client
    async with Client(app) as client:
        yield client

    # Teardown
    if database.engine:
        database.engine.dispose()


@pytest.mark.asyncio
async def test_e2e_workflow(test_client: Client):
    """
    Tests the basic end-to-end workflow.
    """
    # 1. List manuals
    result = await test_client.call_tool("list_manuals")
    assert result.structured_content is not None
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    assert len(manuals) == 1
    manual = manuals[0]
    manual_id = manual.id

    # 2. Get manual metadata
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    assert result.structured_content is not None
    metadata = ManualMetadata.model_validate(result.structured_content)
    # Find Section 1.1
    chapter1 = next(b for b in metadata.table_of_contents if b.title == "Chapter 1")
    section1_1 = next(b for b in chapter1.children if b.title == "Section 1.1")
    bookmark_id = section1_1.id

    # 3. Get markdown content
    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": bookmark_id}
    )
    assert result.structured_content is not None
    content_response = MarkdownContent.model_validate(result.structured_content)
    assert "Content for page 2" in content_response.markdown_content


@pytest.mark.asyncio
async def test_content_retrieval(test_client: Client):
    """
    Tests that get_markdown_content returns full content for a section (and subtree).
    """
    result = await test_client.call_tool("list_manuals")
    manuals = [ManualInfo(**manual) for manual in result.structured_content["result"]]
    manual_id = manuals[0].id
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    metadata = ManualMetadata.model_validate(result.structured_content)

    # Chapter 1 (Pages 1-14). Contains Section 1.1 (Pages 2-14).
    # If we request Chapter 1, we should get content for Page 1 AND Section 1.1 (which is pages 2-14).
    # My manual population logic above assigned Page 1 to Chapter 1, and Page 2-14 to Section 1.1.
    # Section 1.1 is child of Chapter 1.
    # So requesting Chapter 1 should return Page 1 (direct) and Page 2-14 (descendant).

    chapter1 = next(b for b in metadata.table_of_contents if b.title == "Chapter 1")

    result = await test_client.call_tool(
        "get_markdown_content", {"bookmark_id": chapter1.id}
    )
    content = MarkdownContent.model_validate(result.structured_content)

    # Verify we got everything
    assert "Content for page 1." in content.markdown_content
    assert "Content for page 2." in content.markdown_content
    assert "Content for page 5." in content.markdown_content
    assert "Content for page 14." in content.markdown_content

    # Verify no pagination fields (they are removed from model, so accessing them on model would fail or be missing in dict)
    # But result.structured_content is a dict of the returned model.
    # MarkdownContent schema does not have them anymore.


@pytest.mark.asyncio
async def test_search_manual(test_client: Client):
    """
    Tests the search_manual tool using vector search.
    """
    result = await test_client.call_tool("list_manuals")
    manual_id = ManualInfo(**result.structured_content["result"][0]).id

    # Search for something unique to page 2
    query = "Content for page 2"
    result = await test_client.call_tool(
        "search_manual", {"manual_id": manual_id, "query": query}
    )
    search_result = SearchResult.model_validate(result.structured_content)

    # Should find page 2
    match = next(
        (m for m in search_result.results if "Content for page 2" in m.context), None
    )
    assert match is not None
    assert match.manual_id == manual_id

    # Verify hierarchy for Page 2 (Section 1.1 -> Chapter 1)
    # Note: Search results return bookmarks.
    # My population logic assigned Page 2 specifically to Section 1.1.
    # Section 1.1 is child of Chapter 1.
    # The tool constructs the hierarchy.
    assert len(match.bookmarks) >= 2
    titles = [b.title for b in match.bookmarks]
    assert "Chapter 1" in titles
    assert "Section 1.1" in titles


@pytest.mark.asyncio
async def test_search_manual_with_bookmark_filter(test_client: Client):
    """
    Tests the search_manual tool with hierarchical bookmark filtering.
    """
    result = await test_client.call_tool("list_manuals")
    manual_id = ManualInfo(**result.structured_content["result"][0]).id
    result = await test_client.call_tool(
        "get_manual_metadata", {"manual_id": manual_id}
    )
    toc = ManualMetadata.model_validate(result.structured_content).table_of_contents

    chapter1 = next(b for b in toc if b.title == "Chapter 1")
    # Section 1.1 is child
    section1_1 = next(b for b in chapter1.children if b.title == "Section 1.1")

    # Search for "Content for page 10" (which is in Section 1.1)
    # Filter by Chapter 1 (Parent) -> Should match
    query = "Content for page 10"

    result = await test_client.call_tool(
        "search_manual",
        {"manual_id": manual_id, "query": query, "bookmark_id": chapter1.id},
    )
    res = SearchResult.model_validate(result.structured_content)
    assert any("Content for page 10" in m.context for m in res.results)

    # Filter by Section 1.1 (Direct) -> Should match
    result = await test_client.call_tool(
        "search_manual",
        {"manual_id": manual_id, "query": query, "bookmark_id": section1_1.id},
    )
    res = SearchResult.model_validate(result.structured_content)
    assert any("Content for page 10" in m.context for m in res.results)

    # Search for "Content for page 1" (which is in Chapter 1, but NOT Section 1.1)
    # Filter by Section 1.1 -> Should NOT match
    query_p1 = "Content for page 1."  # exact match construction

    result = await test_client.call_tool(
        "search_manual",
        {"manual_id": manual_id, "query": query_p1, "bookmark_id": section1_1.id},
    )
    res = SearchResult.model_validate(result.structured_content)
    # Should be empty or at least not contain page 1
    match_p1 = next(
        (m for m in res.results if "Content for page 1." in m.context), None
    )
    assert match_p1 is None
