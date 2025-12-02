import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from markitdown import MarkItDown
from pydantic import Field
from sqlalchemy.orm import Session, joinedload

from .cache_utils import (
    batch_update_last_accessed,
    create_page_cache,
    find_page_cache,
)
from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Manual
from .pdf_utils import (
    create_temp_pdf_from_page_range,
    search_pdf,
)
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


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server startup event handler."""
    logger.info("Initializing application...")
    # Ensure all necessary directories exist before initializing the database
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()
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


@app.tool(
    name="get_markdown_content",
    description="""Fetches the Markdown content for a specific bookmark (section) within
      a manual. The content is returned in paginated form to handle large sections
      efficiently. Use the `page_offset` and `page_limit` parameters to control
      pagination. The response includes the `next_page_offset` which should be used in
      subsequent calls to retrieve the rest of the content for that section.

    Workflow Example:
    1. Get a `bookmark_id` from the `table_of_contents` provided by
      `get_manual_metadata()`.
    2. Call `get_markdown_content(bookmark_id=...)` to get the first chunk of content.
    3. If `next_page_offset` in the response is not null, call `get_markdown_content()`
      again with the same `bookmark_id` and the new `page_offset` to get the next page.
    4. Repeat until `next_page_offset` is null.""",
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
    page_offset: Annotated[
        int,
        Field(
            description="""The starting page offset within 
            the bookmark section (0-indexed).""",
            default=0,
            ge=0,
        ),
    ] = 0,
    page_limit: Annotated[
        Optional[int],
        Field(
            description="""The maximum number of pages to return.
            Defaults to the server-side setting.""",
            default=None,
            ge=1,
        ),
    ] = None,
) -> MarkdownContent:
    """Returns the Markdown content for a specific bookmark, with pagination."""
    db: Session = SessionLocal()
    markdown_converter = MarkItDown()
    temp_pdf_path: Path | None = None
    try:
        bookmark = (
            db.query(Bookmark)
            .filter(Bookmark.id == bookmark_id)
            .options(joinedload(Bookmark.manual))
            .first()
        )
        if not bookmark:
            raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")

        manual = bookmark.manual
        pdf_path = settings.PDF_ROOT_DIR.resolve() / manual.relative_path

        # 1. Determine the full page range of the bookmark
        bookmark_start_page = bookmark.page_num
        next_bookmark = (
            db.query(Bookmark)
            .filter(
                Bookmark.manual_id == manual.id,
                Bookmark.ordering > bookmark.ordering,
                Bookmark.level <= bookmark.level,
            )
            .order_by(Bookmark.ordering)
            .first()
        )

        if next_bookmark:
            bookmark_end_page = next_bookmark.page_num - 1
        else:
            # If it's the last bookmark, go to the end of the PDF
            bookmark_end_page = manual.page_count

        bookmark_total_pages = (bookmark_end_page - bookmark_start_page) + 1
        if bookmark_total_pages <= 0:
            content_data = {
                "markdown_content": "",
                "bookmark_total_pages": 0,
                "page_offset": 0,
                "page_limit": 0,
                "next_page_offset": None,
            }
            return MarkdownContent.model_validate(content_data)

        # 2. Determine the processing chunk based on limits
        limit = min(
            page_limit or settings.MAX_PAGES_PER_REQUEST, settings.MAX_PAGES_PER_REQUEST
        )

        if page_offset >= bookmark_total_pages:
            raise ToolError("page_offset is out of bounds.")

        # 3. Calculate absolute page numbers for the chunk
        chunk_start_page = bookmark_start_page + page_offset
        chunk_end_page = min(chunk_start_page + limit - 1, bookmark_end_page)

        # 4. Process pages in the chunk
        markdown_parts = []
        processed_page_nums = []
        for page_num in range(chunk_start_page, chunk_end_page + 1):
            processed_page_nums.append(page_num)
            cached_content = find_page_cache(manual, page_num, db)
            if cached_content is not None:
                markdown_parts.append(cached_content)
                continue

            # Cache miss: process the single page
            logger.info(
                f"Cache miss for page {page_num} of '{manual.file_name}'. Processing."
            )
            temp_pdf_path = create_temp_pdf_from_page_range(
                pdf_path, page_num, page_num
            )
            if not temp_pdf_path:
                error_msg = f"""Page {page_num} could not be processed: 
                failed to create temporary PDF."""
                logger.error(error_msg)
                raise ToolError(error_msg)

            try:
                conversion_result = markdown_converter.convert(str(temp_pdf_path))
                page_content = conversion_result.markdown if conversion_result else ""
                if not page_content:
                    logger.warning(
                        f"""Page {page_num} of '{manual.file_name}' 
                        converted to empty content."""
                    )

                markdown_parts.append(page_content)
                create_page_cache(manual, page_num, page_content, db)
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        # 5. Batch update access times
        batch_update_last_accessed(manual.id, processed_page_nums, db)

        # All DB operations for the chunk are complete, commit them.
        db.commit()

        # 6. Construct the response
        final_content = "\n\n---\n\n".join(markdown_parts)
        actual_limit = len(processed_page_nums)
        next_page_offset = page_offset + actual_limit
        if next_page_offset >= bookmark_total_pages:
            next_page_offset = None

        content_data = {
            "markdown_content": final_content,
            "bookmark_total_pages": bookmark_total_pages,
            "page_offset": page_offset,
            "page_limit": actual_limit,
            "next_page_offset": next_page_offset,
        }
        return MarkdownContent.model_validate(content_data)

    except Exception as e:
        db.rollback()
        logger.exception(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


@app.tool(
    name="search_manual",
    description="""Searches for a query string within a specific manual.
    Returns a list of matches, each with the page number, context, and the
    hierarchical path of headings (bookmarks) leading to that page.
    The search is case-insensitive.
    
    Optionally, a `bookmark_id` can be provided to restrict the search to a specific
    section of the manual.

    Workflow Example:
    1. Call `search_manual(manual_id=..., query="...")` to find occurrences.
    2. Use the `page_offset` from a result item to jump to that location using
       `get_markdown_content`.
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
        Field(description="Optional bookmark ID to restrict search to a specific section."),
    ] = None,
) -> SearchResult:
    """Searches for text in a manual and returns matches with context and hierarchy."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.id == manual_id).first()
        if not manual:
            raise ToolError(f"Manual with id '{manual_id}' not found.")

        pdf_path = settings.PDF_ROOT_DIR.resolve() / manual.relative_path
        
        start_page = None
        end_page = None
        
        if bookmark_id:
            bookmark = (
                db.query(Bookmark)
                .filter(Bookmark.id == bookmark_id)
                .first()
            )
            if not bookmark:
                raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")
            
            if bookmark.manual_id != manual_id:
                raise ToolError(f"Bookmark '{bookmark_id}' does not belong to manual '{manual_id}'.")
            
            start_page = bookmark.page_num
            
            # Find the next bookmark to determine the end of this section
            next_bookmark = (
                db.query(Bookmark)
                .filter(
                    Bookmark.manual_id == manual.id,
                    Bookmark.ordering > bookmark.ordering,
                    Bookmark.level <= bookmark.level,
                )
                .order_by(Bookmark.ordering)
                .first()
            )
            
            if next_bookmark:
                end_page = next_bookmark.page_num - 1
            else:
                end_page = manual.page_count

        # 1. Perform the text search on the PDF
        pdf_matches = search_pdf(pdf_path, query, start_page=start_page, end_page=end_page)
        
        # 2. Enhance matches with bookmark hierarchy
        results = []
        
        # Pre-fetch all bookmarks for this manual to minimize DB queries
        # Ordered by page_num then ordering to help with finding the right bookmark
        all_bookmarks = (
            db.query(Bookmark)
            .filter(Bookmark.manual_id == manual.id)
            .order_by(Bookmark.page_num.asc(), Bookmark.ordering.asc())
            .all()
        )
        
        for match in pdf_matches:
            # Find the deepest bookmark that starts on or before the match page
            # Since bookmarks are ordered by page_num, we can iterate to find the best candidate
            current_bookmark = None
            for bm in all_bookmarks:
                if bm.page_num <= match.page_num:
                    current_bookmark = bm
                else:
                    # We've passed the possible bookmarks for this page
                    break
            
            # Build hierarchy
            hierarchy = []
            bookmark_node_list = []
            page_offset = 0
            
            if current_bookmark:
                # Calculate offset from the start of the bookmark
                page_offset = match.page_num - current_bookmark.page_num
                
                # Traverse up to build the full hierarchy
                # We need to reconstruct the path. Since we have the parent_id, we can do this.
                # However, our Bookmark model has a parent relationship, so we can use that if loaded.
                # But we didn't eager load parents in the bulk query.
                # Let's just use a map for O(1) lookup since we have all bookmarks.
                bookmark_map = {bm.id: bm for bm in all_bookmarks}
                
                temp_bm = current_bookmark
                path_nodes = []
                while temp_bm:
                    path_nodes.append(temp_bm)
                    if temp_bm.parent_id:
                        temp_bm = bookmark_map.get(temp_bm.parent_id)
                    else:
                        temp_bm = None
                
                # Reverse to get root-to-leaf order
                path_nodes.reverse()
                
                # Convert to BookmarkNode (simplified, no children needed for the path list itself usually, 
                # but schema asks for BookmarkNode which has children field. 
                # We will return the nodes as a flat list representing the path, 
                # so 'children' can be empty for these nodes in this context.)
                for node in path_nodes:
                    bookmark_node_list.append(BookmarkNode(
                        id=node.id,
                        title=node.title,
                        page=node.page_num,
                        children=[] # We don't need the full tree here, just the path
                    ))
            else:
                 # Match is before any bookmark (e.g. title page)
                 # We can leave bookmarks list empty or create a pseudo-node if needed.
                 # Schema says bookmarks is List[BookmarkNode]. Empty list is valid.
                 page_offset = match.page_num - 1 # Offset from start of doc if no bookmark
            
            results.append(SearchResultItem(
                page_num=match.page_num,
                page_offset=page_offset,
                bookmarks=bookmark_node_list,
                context=match.context,
                match_index=match.match_index
            ))
            
        return SearchResult(
            manual_id=manual_id,
            query=query,
            results=results
        )

    except Exception as e:
        logger.error(f"Error searching manual '{manual_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


if __name__ == "__main__":
    app.run(transport="http", host=settings.HOST, port=settings.PORT)
