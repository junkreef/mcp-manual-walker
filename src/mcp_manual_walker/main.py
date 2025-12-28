import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Manual
from .schemas import (
    BookmarkNode,
    ManualInfo,
    ManualMetadata,
    MarkdownContent,
    SearchResult,
    SearchResultItem,
)

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global ChromaDB client and collection
chroma_client = None
collection = None


def get_embedding_function():
    # Helper to get the embedding function
    # Using intfloat/multilingual-e5-small as configured in builder
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server startup event handler."""
    logger.info("Initializing application...")
    # Ensure all necessary directories exist before initializing the database
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMADB_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()

    logger.info("Initializing ChromaDB...")
    global chroma_client, collection
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMADB_PATH.resolve())
        )
        embedding_fn = get_embedding_function()
        # Expect collection to exist from builder
        collection = chroma_client.get_collection(
            name="manual_chunks", embedding_function=embedding_fn
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        # We don't raise here to allow server to start, but tools will fail if not fixed.

    yield


app = FastMCP(lifespan=lifespan)


@app.tool(
    name="list_manuals",
    description="""Provides a comprehensive list of all available manuals. This tool is
    the primary entry point for discovering content. It returns a list of all manuals
    found in the system, each with a unique ID that is required by other tools
    like `get_manual_metadata`.

    Workflow Example:
    1. Call `list_manuals()` to get a list of all available manuals.
    2. Identify the manual you are interested in from the list.
    3. Use the `id` of that manual to call `get_manual_metadata()` to retrieve its
       table of contents and other details.""",
    tags={"manual", "discovery"},
    annotations={"readOnlyHint": True},
)
def list_manuals() -> List[ManualInfo]:
    """Returns a list of all available manuals."""
    db: Session = SessionLocal()
    try:
        manuals = db.query(Manual).order_by(Manual.file_name).all()
        return [
            ManualInfo(
                id=m.id,
                file_name=m.file_name,
                document_title=m.document_title,
            )
            for m in manuals
        ]
    except Exception as e:
        logger.error(f"Error fetching list of manuals: {e}")
        raise ToolError(e)
    finally:
        db.close()


def _build_toc(bookmarks: list[Bookmark]) -> list[BookmarkNode]:
    """Builds a nested table of contents from a flat list of bookmarks."""
    toc = []
    bookmark_map = {
        bm.id: BookmarkNode(id=bm.id, title=bm.title, page=bm.page_num, children=[])
        for bm in bookmarks
    }
    for bm in bookmarks:
        if bm.parent_id:
            if parent := bookmark_map.get(bm.parent_id):
                parent.children.append(bookmark_map[bm.id])
        else:
            toc.append(bookmark_map[bm.id])
    return toc


@app.tool(
    name="get_manual_metadata",
    description="""Retrieves detailed metadata and a hierarchical table of contents for
    a specific manual. Use this tool after you have identified a manual of interest
    using `list_manuals()`. It provides the full structure of the manual's bookmarks,
    which is essential for navigating its content. Each bookmark in the table of
    contents has its own unique ID, which is required by the `get_markdown_content`
    tool to fetch the actual content of that section.

    Workflow Example:
    1. Get a `manual_id` from the output of `list_manuals()`.
    2. Call `get_manual_metadata(manual_id=...)` to get the manual's structure.
    3. Browse the `table_of_contents` to find the specific section you need.
    4. Use the `id` of the desired bookmark to call `get_markdown_content()`.""",
    tags={"manual", "metadata", "toc"},
    annotations={"readOnlyHint": True},
)
def get_manual_metadata(
    manual_id: Annotated[
        str,
        Field(
            description="""The unique ID of the manual, 
            obtained from the `list_manuals` tool."""
        ),
    ],
) -> ManualMetadata:
    """Returns metadata and a hierarchical table of contents for a specified manual."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.id == manual_id).first()
        if not manual:
            raise ToolError(f"Manual with id '{manual_id}' not found.")

        bookmarks = (
            db.query(Bookmark)
            .filter(Bookmark.manual_id == manual.id)
            .order_by(Bookmark.ordering)
            .all()
        )
        table_of_contents = _build_toc(bookmarks)

        manual_data = {
            "id": manual.id,
            "file_name": manual.file_name,
            "document_title": manual.document_title,
            "file_hash": manual.file_hash,
            "table_of_contents": table_of_contents,
        }
        return ManualMetadata.model_validate(manual_data)
    except Exception as e:
        logger.error(f"Error fetching metadata for manual_id '{manual_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


def _get_descendant_bookmark_ids(
    manual_id: str, bookmark_id: str, db: Session
) -> List[str]:
    """Retrieves the list of bookmark IDs for the given bookmark and all its descendants."""
    target_bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
    if not target_bookmark:
        raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")

    if target_bookmark.manual_id != manual_id:
        raise ToolError(
            f"Bookmark '{bookmark_id}' does not belong to manual '{manual_id}'."
        )

    # Efficiently find descendants.
    # Since we have 'ordering' and 'level', descendants follow immediately
    # and have level > target.level.
    # We stop when we hit a bookmark with level <= target.level.

    # Get all subsequent bookmarks for this manual
    subsequent_bookmarks = (
        db.query(Bookmark)
        .filter(
            Bookmark.manual_id == manual_id,
            Bookmark.ordering >= target_bookmark.ordering,
        )
        .order_by(Bookmark.ordering)
        .all()
    )

    descendant_ids = []
    # The first one is the target itself
    for bm in subsequent_bookmarks:
        if bm.id == bookmark_id:
            descendant_ids.append(bm.id)
            continue

        if bm.level > target_bookmark.level:
            descendant_ids.append(bm.id)
        else:
            # We reached a sibling or parent (level <= target), so we stop
            break

    return descendant_ids


@app.tool(
    name="get_markdown_content",
    description="""Fetches the Markdown content for a specific bookmark (section) within
      a manual using the Vector DB. This returns the pre-processed text chunks associated
      with the bookmark and its sub-sections.
      
    Workflow Example:
    1. Get a `bookmark_id` from the `table_of_contents` provided by
      `get_manual_metadata()`.
    2. Call `get_markdown_content(bookmark_id=...)` to get the content.""",
    tags={"manual", "content", "markdown"},
    annotations={"readOnlyHint": True},
)
def get_markdown_content(
    bookmark_id: Annotated[
        str,
        Field(
            description="""The unique ID of the bookmark, 
            obtained from `get_manual_metadata`."""
        ),
    ],
) -> MarkdownContent:
    """Returns the Markdown content for a specific bookmark from the Vector DB."""
    if collection is None:
        raise ToolError("Vector database is not initialized.")

    db: Session = SessionLocal()
    try:
        # Resolve bookmark and manual
        bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
        if not bookmark:
            raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")

        manual_id = bookmark.manual_id

        # Get all relevant bookmark IDs (hierarchical)
        target_bookmark_ids = _get_descendant_bookmark_ids(manual_id, bookmark_id, db)

        # Query ChromaDB
        # We want chunks where manual_id matches AND bookmark_id is in our list

        # Chroma where clause:
        # {"$and": [{"manual_id": manual_id}, {"bookmark_id": {"$in": target_bookmark_ids}}]}

        results = collection.get(
            where={
                "$and": [
                    {"manual_id": manual_id},
                    {"bookmark_id": {"$in": target_bookmark_ids}},
                ]
            }
        )

        # results['documents'] is a list of strings
        # results['ids'] is a list of IDs.
        # We should sort them. Assuming IDs are like "manual_id_index" where index is integer.
        # But ids are strings. "uuid_0", "uuid_1", "uuid_10". String sort is bad for "10" vs "2".
        # Let's try to extract index from ID.

        # Safely zip and sort
        combined = []
        if results["ids"] and results["documents"]:
            for i, doc_id in enumerate(results["ids"]):
                # doc_id format expected: "{manual_id}_{index}"
                parts = doc_id.rsplit("_", 1)
                idx = 0
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1])
                elif len(parts) == 2:
                    # fallback if manual_id contains _
                    pass

                combined.append((idx, results["documents"][i]))

        # Sort by index
        combined.sort(key=lambda x: x[0])

        # Join text
        avg_text = [t for _, t in combined]
        final_content = "\n\n".join(avg_text)

        return MarkdownContent(markdown_content=final_content)

    except Exception as e:
        logger.exception(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


@app.tool(
    name="search_manual",
    description="""Searches for a query string within a specific manual using semantic search.
    Returns the top matching text chunks.
    
    Optionally, a `bookmark_id` can be provided to restrict the search to a specific
    section of the manual (including subsections).

    Workflow Example:
    1. Call `search_manual(manual_id=..., query="...")` to find occurrences.
    """,
    tags={"manual", "search"},
    annotations={"readOnlyHint": True},
)
def search_manual(
    manual_id: Annotated[
        str,
        Field(description="The unique ID of the manual to search."),
    ],
    query: Annotated[
        str,
        Field(description="The text to search for."),
    ],
    bookmark_id: Annotated[
        Optional[str],
        Field(
            description="Optional bookmark ID to restrict search to a specific section."
        ),
    ] = None,
) -> SearchResult:
    """Searches for text in a manual and returns matches with context and hierarchy."""
    if collection is None:
        raise ToolError("Vector database is not initialized.")

    db: Session = SessionLocal()
    try:
        where_clause = {"manual_id": manual_id}

        if bookmark_id:
            # Hierarchical filter
            target_ids = _get_descendant_bookmark_ids(manual_id, bookmark_id, db)
            where_clause = {
                "$and": [{"manual_id": manual_id}, {"bookmark_id": {"$in": target_ids}}]
            }

        # Query ChromaDB
        results = collection.query(query_texts=[query], n_results=5, where=where_clause)

        # results is a dict with lists of lists (for documents, metadatas, etc.)
        # Structure: {'ids': [['id1', ...]], 'metadatas': [[{...}, ...]], 'documents': [['text', ...]]}

        search_result_items = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            # Pre-fetch bookmarks for hierarchy reconstruction
            # We can't easily pre-fetch just the parents needed without knowing them.
            # But we can cache the manual's bookmarks map.
            all_bookmarks = (
                db.query(Bookmark).filter(Bookmark.manual_id == manual_id).all()
            )
            bookmark_map = {bm.id: bm for bm in all_bookmarks}

            for i, chunk_id in enumerate(ids):
                text = docs[i]
                meta = metas[i]

                # Get Bookmark Info
                chunk_bm_id = meta.get("bookmark_id")

                # Build hierarchy path
                bookmark_node_list = []

                if chunk_bm_id and chunk_bm_id in bookmark_map:
                    temp_bm = bookmark_map[chunk_bm_id]
                    path_nodes = []
                    while temp_bm:
                        path_nodes.append(temp_bm)
                        if temp_bm.parent_id:
                            temp_bm = bookmark_map.get(temp_bm.parent_id)
                        else:
                            temp_bm = None
                    path_nodes.reverse()

                    for node in path_nodes:
                        bookmark_node_list.append(
                            BookmarkNode(
                                id=node.id,
                                title=node.title,
                                page=node.page_num,
                                children=[],
                            )
                        )

                search_result_items.append(
                    SearchResultItem(
                        bookmarks=bookmark_node_list,
                        context=text,
                        manual_id=manual_id,
                        bookmark_id=chunk_bm_id,
                    )
                )

        return SearchResult(
            manual_id=manual_id, query=query, results=search_result_items
        )

    except Exception as e:
        logger.error(f"Error searching manual '{manual_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


if __name__ == "__main__":
    app.run(transport="http", host=settings.HOST, port=settings.PORT)
