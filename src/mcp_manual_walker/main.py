import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from markitdown import MarkItDown
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


@app.tool()
def list_manuals() -> list[dict]:
    """Returns a list of all available manuals, including their ID, filename, and title."""
    db: Session = SessionLocal()
    try:
        manuals = db.query(Manual).order_by(Manual.file_name).all()
        return [
            {
                "id": m.id,
                "file_name": m.file_name,
                "document_title": m.document_title,
            }
            for m in manuals
        ]
    except Exception as e:
        logger.error(f"Error fetching list of manuals: {e}")
        return []
    finally:
        db.close()


def _build_toc(bookmarks: list[Bookmark]) -> list[dict]:
    """Builds a nested table of contents from a flat list of bookmarks."""
    toc = []
    bookmark_map = {
        bm.id: {"id": bm.id, "title": bm.title, "page": bm.page_num, "children": []}
        for bm in bookmarks
    }
    for bm in bookmarks:
        if bm.parent_id:
            if parent := bookmark_map.get(bm.parent_id):
                parent["children"].append(bookmark_map[bm.id])
        else:
            toc.append(bookmark_map[bm.id])
    return toc


@app.tool()
def get_manual_metadata(manual_id: str) -> dict:
    """Returns metadata and hierarchical bookmark info for a specified manual."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.id == manual_id).first()
        if not manual:
            return {"error": f"Manual with id '{manual_id}' not found."}

        bookmarks = (
            db.query(Bookmark)
            .filter(Bookmark.manual_id == manual.id)
            .order_by(Bookmark.ordering)
            .all()
        )
        table_of_contents = _build_toc(bookmarks)

        return {
            "id": manual.id,
            "file_name": manual.file_name,
            "document_title": manual.document_title,
            "file_hash": manual.file_hash,
            "table_of_contents": table_of_contents,
        }
    except Exception as e:
        logger.error(f"Error fetching metadata for manual_id '{manual_id}': {e}")
        return {"error": "An internal error occurred."}
    finally:
        db.close()


@app.tool()
def get_markdown_content(
    bookmark_id: str, page_offset: int = 0, page_limit: Optional[int] = None
) -> dict:
    """
    Returns the Markdown content for a specific bookmark, with pagination.
    """
    db: Session = SessionLocal()
    markdown_converter = MarkItDown()
    temp_pdf_path: Path | None = None
    try:
        bookmark = (
            db.query(Bookmark).filter(Bookmark.id == bookmark_id).options(joinedload(Bookmark.manual)).first()
        )
        if not bookmark:
            return {"error": f"Bookmark with id '{bookmark_id}' not found."}

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
            return {
                "markdown_content": "",
                "bookmark_total_pages": 0,
                "page_offset": 0,
                "page_limit": 0,
                "next_page_offset": None,
            }

        # 2. Determine the processing chunk based on limits
        limit = min(page_limit or settings.MAX_PAGES_PER_REQUEST, settings.MAX_PAGES_PER_REQUEST)
        
        if page_offset >= bookmark_total_pages:
            return {"error": "page_offset is out of bounds."}

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
            logger.info(f"Cache miss for page {page_num} of '{manual.file_name}'. Processing.")
            temp_pdf_path = create_temp_pdf_from_page_range(pdf_path, page_num, page_num)
            if not temp_pdf_path:
                error_msg = f"Page {page_num} could not be processed: failed to create temporary PDF."
                logger.error(error_msg)
                return {"error": error_msg}

            try:
                conversion_result = markdown_converter.convert(str(temp_pdf_path))
                page_content = conversion_result.markdown if conversion_result else ""
                if not page_content:
                    logger.warning(f"Page {page_num} of '{manual.file_name}' converted to empty content.")
                
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

        return {
            "markdown_content": final_content,
            "bookmark_total_pages": bookmark_total_pages,
            "page_offset": page_offset,
            "page_limit": actual_limit,
            "next_page_offset": next_page_offset,
        }

    except Exception as e:
        db.rollback()
        logger.exception(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        return {"error": "An internal error occurred while fetching content."}
    finally:
        db.close()


if __name__ == "__main__":
    app.run(transport="http", host=settings.HOST, port=settings.PORT)