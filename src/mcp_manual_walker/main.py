import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP
from markitdown import MarkItDown
from sqlalchemy.orm import Session, joinedload

from .cache_utils import create_cache, find_valid_cache
from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Cache, Manual
from .pdf_utils import (
    calculate_file_hash,
    create_temp_pdf_from_page_range,
    extract_pdf_metadata,
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
                for i, bm_data in enumerate(pdf_data["bookmarks"]):
                    level = bm_data["level"]
                    parent = parent_stack.get(level - 1)
                    bookmark = Bookmark(
                        manual_id=new_manual.id,
                        ordering=i,
                        title=bm_data["title"],
                        level=level,
                        page_num=bm_data["page_num"],
                        parent_id=parent.id if parent else None,
                    )
                    db.add(bookmark)
                    db.flush()  # Flush to get the new bookmark's ID
                    parent_stack[level] = bookmark
                    keys_to_del = [k for k in parent_stack if k > level]
                    for k in keys_to_del:
                        del parent_stack[k]
                logger.info(f"Successfully processed and added '{relative_path}'.")

        deleted_paths = db_paths - fs_paths
        for path_to_delete in deleted_paths:
            manual_to_delete = db_manuals[path_to_delete]
            logger.info(f"'{path_to_delete}' has been removed. Deleting from database.")

            # Query for all cache entries related to the manual being deleted
            cache_files_to_delete = (
                db.query(Cache.markdown_file_path)
                .join(Bookmark)
                .filter(Bookmark.manual_id == manual_to_delete.id)
                .all()
            )

            # Delete the physical cache files
            for cache_file in cache_files_to_delete:
                try:
                    file_path = Path(cache_file[0])
                    if file_path.is_file():
                        os.remove(file_path)
                        logger.info(f"Deleted orphaned cache file: {file_path}")
                except FileNotFoundError:
                    logger.warning(
                        f"Cache file not found, skipping deletion: {cache_file[0]}"
                    )
                except Exception as e:
                    logger.error(f"Error deleting cache file {cache_file[0]}: {e}")

            db.delete(manual_to_delete)

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
    logger.info("Initializing application...")
    # Ensure all necessary directories exist before initializing the database
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()
    sync_database()
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
def get_markdown_content(bookmark_id: str) -> str:
    """
    Returns the Markdown content for a specific bookmark.
    Uses a cache to speed up repeated requests.
    """
    db: Session = SessionLocal()
    markdown_converter = MarkItDown()
    temp_pdf_path: Path | None = None
    try:
        bookmark = (
            db.query(Bookmark)
            .filter(Bookmark.id == bookmark_id)
            .options(joinedload(Bookmark.manual), joinedload(Bookmark.cache_entry))
            .first()
        )
        if not bookmark:
            return f"Error: Bookmark with id '{bookmark_id}' not found."

        if cached_content := find_valid_cache(bookmark, db):
            return cached_content

        manual = bookmark.manual
        pdf_path = settings.PDF_ROOT_DIR.resolve() / manual.relative_path
        start_page = bookmark.page_num

        # Find the next bookmark at the same or higher level to determine the end page.
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

        end_page = (
            next_bookmark.page_num - 1
            if next_bookmark and next_bookmark.page_num > start_page
            else None
        )

        logger.info(
            f"Cache miss for bookmark '{bookmark.title}'. "
            f"Creating temporary PDF for pages {start_page}-{end_page or 'end'} from '{pdf_path.name}'."
        )

        temp_pdf_path = create_temp_pdf_from_page_range(pdf_path, start_page, end_page)
        if not temp_pdf_path:
            return f"Error: Could not create temporary PDF for bookmark '{bookmark.title}'."

        markdown_content = markdown_converter.convert(str(temp_pdf_path))
        if not markdown_content:
            return f"Error: Failed to convert content for bookmark '{bookmark.title}' to Markdown."

        create_cache(bookmark, markdown_content.markdown, db)
        return markdown_content.markdown

    except Exception as e:
        logger.error(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        return "Error: An internal error occurred while fetching content."
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info(f"Successfully removed temporary file: {temp_pdf_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_pdf_path}: {e}")
        db.close()


if __name__ == "__main__":
    app.run(transport="http", host=settings.HOST, port=settings.PORT)