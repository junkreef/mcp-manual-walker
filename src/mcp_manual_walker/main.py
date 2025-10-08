import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastmcp import FastMCP
from markitdown import MarkItDown
from sqlalchemy.orm import Session, joinedload

from .cache_utils import create_cache, find_valid_cache
from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Manual
from .pdf_utils import (
    calculate_file_hash,
    extract_pdf_metadata,
    extract_text_from_page_range,
    scan_pdfs,
)

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def sync_database():
    """
    Scans the PDF directory and synchronizes the database.
    - Adds new manuals.
    - Updates manuals that have changed.
    - Deletes manuals that are no longer on the filesystem.
    """
    logger.info("Starting database synchronization...")
    db: Session = SessionLocal()
    try:
        db_manuals = {m.relative_path: m for m in db.query(Manual).all()}
        db_paths = set(db_manuals.keys())

        fs_paths = set()
        pdf_root_path = settings.PDF_ROOT_DIR.resolve()

        for pdf_path in scan_pdfs(pdf_root_path):
            relative_path = str(pdf_path.relative_to(pdf_root_path))
            fs_paths.add(relative_path)

            try:
                file_hash = calculate_file_hash(pdf_path)
            except IOError as e:
                logger.error(f"Could not calculate hash for {pdf_path}: {e}")
                continue

            if relative_path not in db_paths or db_manuals[relative_path].file_hash != file_hash:
                if relative_path in db_paths:
                    logger.info(f"'{relative_path}' has been updated. Re-processing.")
                    db.delete(db_manuals[relative_path])
                    db.commit()
                else:
                    logger.info(f"New manual found: '{relative_path}'")

                pdf_data = extract_pdf_metadata(pdf_path)
                if not pdf_data:
                    logger.warning(f"Could not extract metadata from {pdf_path}. Skipping.")
                    continue

                new_manual = Manual(
                    file_name=pdf_path.name,
                    document_title=pdf_data["document_title"],
                    relative_path=relative_path,
                    file_hash=file_hash,
                )
                db.add(new_manual)
                db.flush()

                parent_stack = {}
                for bm_data in pdf_data["bookmarks"]:
                    level = bm_data["level"]
                    parent = parent_stack.get(level - 1)
                    bookmark = Bookmark(
                        manual_id=new_manual.id,
                        title=bm_data["title"],
                        level=level,
                        page_num=bm_data["page_num"],
                        parent_id=parent.id if parent else None,
                    )
                    db.add(bookmark)
                    parent_stack[level] = bookmark
                    keys_to_del = [k for k in parent_stack if k > level]
                    for k in keys_to_del:
                        del parent_stack[k]
                logger.info(f"Successfully processed and added '{relative_path}'.")

        deleted_paths = db_paths - fs_paths
        for path_to_delete in deleted_paths:
            logger.info(f"'{path_to_delete}' has been removed. Deleting from database.")
            db.delete(db_manuals[path_to_delete])

        db.commit()

    except Exception as e:
        logger.error(f"An error occurred during database synchronization: {e}")
        db.rollback()
    finally:
        db.close()
    logger.info("Database synchronization complete.")


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server startup event handler."""
    logger.info("Initializing database...")
    init_db()
    settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sync_database()
    yield


app = FastMCP(lifespan=lifespan)


@app.tool()
def list_manuals() -> list[dict[str, str]]:
    """Returns a list of all available manuals, including their filenames and document titles."""
    db: Session = SessionLocal()
    try:
        manuals = db.query(Manual).order_by(Manual.file_name).all()
        return [{"file_name": m.file_name, "document_title": m.document_title} for m in manuals]
    except Exception as e:
        logger.error(f"Error fetching list of manuals: {e}")
        return []
    finally:
        db.close()


def _build_toc(bookmarks: list[Bookmark]) -> list[dict]:
    """Builds a nested table of contents from a flat list of bookmarks."""
    toc = []
    bookmark_map = {bm.id: {"title": bm.title, "page": bm.page_num, "children": []} for bm in bookmarks}
    for bm in bookmarks:
        if bm.parent_id:
            if parent := bookmark_map.get(bm.parent_id):
                parent["children"].append(bookmark_map[bm.id])
        else:
            toc.append(bookmark_map[bm.id])
    return toc


@app.tool()
def get_manual_metadata(file_name: str) -> dict:
    """Returns metadata and hierarchical bookmark information for a specified manual."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.file_name == file_name).first()
        if not manual:
            return {"error": f"Manual with file_name '{file_name}' not found."}

        bookmarks = db.query(Bookmark).filter(Bookmark.manual_id == manual.id).order_by(Bookmark.id).all()
        table_of_contents = _build_toc(bookmarks)

        return {
            "file_name": manual.file_name,
            "document_title": manual.document_title,
            "file_hash": manual.file_hash,
            "table_of_contents": table_of_contents,
        }
    except Exception as e:
        logger.error(f"Error fetching metadata for '{file_name}': {e}")
        return {"error": "An internal error occurred."}
    finally:
        db.close()


@app.tool()
def get_markdown_content(file_name: str, bookmark_title: str) -> str:
    """
    Returns the Markdown content for a specific bookmark within a specified manual.
    It uses a cache to speed up repeated requests.
    """
    db: Session = SessionLocal()
    markdown_converter = MarkItDown()
    try:
        bookmark = (
            db.query(Bookmark)
            .join(Manual)
            .filter(Manual.file_name == file_name, Bookmark.title == bookmark_title)
            .options(joinedload(Bookmark.manual), joinedload(Bookmark.cache_entry))
            .first()
        )
        if not bookmark:
            return f"Error: Bookmark '{bookmark_title}' not found in manual '{file_name}'."

        if cached_content := find_valid_cache(bookmark, db):
            return cached_content

        manual = bookmark.manual
        pdf_path = settings.PDF_ROOT_DIR.resolve() / manual.relative_path
        start_page = bookmark.page_num

        next_bookmark = (
            db.query(Bookmark)
            .filter(Bookmark.manual_id == manual.id, Bookmark.level <= bookmark.level, Bookmark.id > bookmark.id)
            .order_by(Bookmark.id)
            .first()
        )
        end_page = next_bookmark.page_num - 1 if next_bookmark and next_bookmark.page_num > start_page else None

        logger.info(f"Cache miss. Extracting pages {start_page}-{end_page or 'end'} from '{pdf_path.name}'.")
        raw_text = extract_text_from_page_range(pdf_path, start_page, end_page)
        if not raw_text:
            return f"Error: Could not extract text for bookmark '{bookmark_title}'."

        markdown_content = markdown_converter.convert(raw_text)
        create_cache(bookmark, markdown_content, db)
        return markdown_content

    except Exception as e:
        logger.error(f"Error getting content for '{bookmark_title}' in '{file_name}': {e}")
        return "Error: An internal error occurred while fetching content."
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)