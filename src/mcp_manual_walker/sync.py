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

    If a db session is provided, it will be used. Otherwise, a new session
    will be created and managed internally.
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
                pdf_data = extract_pdf_metadata(pdf_path)
                if not pdf_data:
                    logger.warning(
                        f"Could not extract metadata from {pdf_path}. Skipping."
                    )
                    continue

                # Create a new manual instance to replace the old one or for a new file
                new_manual = Manual(
                    file_name=pdf_path.name,
                    document_title=pdf_data["document_title"],
                    relative_path=relative_path,
                    file_hash=file_hash,
                    page_count=pdf_data["page_count"],
                )

                manual_to_update = db_manuals.get(relative_path)
                if manual_to_update:
                    logger.info(
                        f"'{relative_path}' has been updated. Updating existing manual."
                    )
                    # Since bookmarks are re-added, clear old ones.
                    # Cascade delete handles this.
                    manual_to_update.file_hash = file_hash
                else:
                    session.add(new_manual)

                session.flush()  # Flush to get the new_manual.id for bookmark relations

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
                    session.add(bookmark)
                    session.flush()  # Necessary to get parent IDs for nested bookmarks
                    parent_stack[level] = bookmark
                    keys_to_del = [k for k in parent_stack if k > level]
                    for k in keys_to_del:
                        del parent_stack[k]
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
