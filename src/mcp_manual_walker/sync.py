import logging
import shutil

from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Manual
from .pdf_utils import (
    calculate_file_hash,
    extract_pdf_metadata,
    scan_pdfs,
)

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def sync_database() -> None:
    """
    Scans the PDF directory and synchronizes the database.
    - Adds new manuals.
    - Updates manuals that have changed.
    - Deletes manuals that are no longer on the filesystem.
    """
    logger.info("Starting database synchronization...")
    init_db()

    session = SessionLocal()

    try:
        db_manuals = {m.relative_path: m for m in session.query(Manual).all()}
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

            if (
                relative_path not in db_paths
                or db_manuals[relative_path].file_hash != file_hash
            ):
                # An existing manual has been updated, or this is a new manual.
                # In both cases, we'll delete the old one (if it exists) and create a new one.
                # This simplifies logic and ensures bookmarks and cache are correctly handled.
                manual_to_delete = db_manuals.get(relative_path)
                if manual_to_delete:
                    logger.info(f"'{relative_path}' has been updated. Re-processing.")
                    # Delete old cache files on disk first
                    shutil.rmtree(settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True)
                    # Deletion of the manual object will cascade in the DB
                    session.delete(manual_to_delete)
                    # We need to flush to ensure the delete is processed before we add a new
                    # manual with potentially the same unique constraints.
                    session.flush()
                else:
                    logger.info(f"New manual found: '{relative_path}'")

                pdf_data = extract_pdf_metadata(pdf_path)
                if not pdf_data:
                    logger.warning(
                        f"Could not extract metadata from {pdf_path}. Skipping."
                    )
                    continue

                new_manual = Manual(
                    file_name=pdf_path.name,
                    document_title=pdf_data["document_title"],
                    relative_path=relative_path,
                    file_hash=file_hash,
                    page_count=pdf_data["page_count"],
                )
                session.add(new_manual)

                parent_stack = {}
                bookmarks_to_add = []
                for i, bm_data in enumerate(pdf_data["bookmarks"]):
                    level = bm_data["level"]
                    parent = parent_stack.get(level - 1)
                    bookmark = Bookmark(
                        manual=new_manual,
                        ordering=i,
                        title=bm_data["title"],
                        level=level,
                        page_num=bm_data["page_num"],
                        parent=parent,
                    )
                    bookmarks_to_add.append(bookmark)
                    parent_stack[level] = bookmark
                    keys_to_del = [k for k in parent_stack if k > level]
                    for k in keys_to_del:
                        del parent_stack[k]
                session.add_all(bookmarks_to_add)
                session.flush()
                logger.info(f"Successfully processed and added '{relative_path}'.")

        deleted_paths = db_paths - fs_paths
        for path_to_delete in deleted_paths:
            manual_to_delete = db_manuals[path_to_delete]
            logger.info(f"'{path_to_delete}' has been removed. Deleting from database.")
            shutil.rmtree(settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True)
            session.delete(manual_to_delete)

        session.commit()

    except Exception as e:
        logger.error(f"An error occurred during database synchronization: {e}")
        session.rollback()
        raise  # Re-raise exception to make caller aware of the failure
    finally:
        session.close()

    logger.info("Database synchronization complete.")


if __name__ == "__main__":
    sync_database()
