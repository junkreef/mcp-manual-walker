import logging
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, TypedDict

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


# Type definition for the result of PDF processing
class PDFProcessResult(TypedDict):
    relative_path: str
    file_hash: Optional[str]
    pdf_data: Optional[dict]
    error: Optional[str]


def process_pdf_file(pdf_path_dict: dict) -> PDFProcessResult:
    """
    Processes a single PDF file to extract its hash and metadata.
    Designed to be run in a separate process.
    """
    
    relative_path = pdf_path_dict['relative_path']
    pdf_path = pdf_path_dict['data_root'] / relative_path

    try:
        file_hash = calculate_file_hash(pdf_path)
        if file_hash != pdf_path_dict['current_hash']:
            pdf_data = extract_pdf_metadata(pdf_path)
            return {
                "relative_path": relative_path,
                "file_hash": file_hash,
                "pdf_data": pdf_data,
                "error": None,
            }
        else:
            return {
                "relative_path": relative_path,
                "file_hash": file_hash,
                "pdf_data": None,
                "error": None,
            }

    except Exception as e:
        logger.error(f"Failed to process {relative_path}: {e}")
        return {
            "relative_path": relative_path,
            "file_hash": None,
            "pdf_data": None,
            "error": str(e),
        }


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

        pdf_root_path = settings.PDF_ROOT_DIR.resolve()
        all_pdf_paths = [{
            'data_root': pdf_root_path, 
            'relative_path': str(p.relative_to(pdf_root_path)),
            'current_hash':
                db_manuals[str(p.relative_to(pdf_root_path))].file_hash
                if db_manuals.get(str(p.relative_to(pdf_root_path)))
                else None
            } for p in scan_pdfs(pdf_root_path)]

        fs_paths: set[str] = set()
        processed_results: list[PDFProcessResult] = []

        # Use ProcessPoolExecutor for parallel PDF processing
        with ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as executor:
            # Map process_pdf_file to all PDF paths and collect results
            for result in executor.map(process_pdf_file, all_pdf_paths):
                processed_results.append(result)

        # Process results and synchronize database in the main thread
        for result in processed_results:
            relative_path = result["relative_path"]

            # If the result indicates error, the database record will be removed
            # because it will be treat as deleted by missing records in fs_paths.
            if result["error"]:
                logger.error(
                    f"""Skipping '{relative_path}' due to processing error: 
                    {result['error']}"""
                )
                continue
            
            fs_paths.add(relative_path)

            file_hash = result["file_hash"]
            pdf_data = result["pdf_data"]

            # Check if manual needs to be added or updated
            if (
                relative_path not in db_paths
                or db_manuals[relative_path].file_hash != file_hash
            ):
                manual_to_delete = db_manuals.get(relative_path)
                if manual_to_delete:
                    logger.info(f"'{relative_path}' has been updated. Re-processing.")
                    # Delete old cache files on disk first
                    shutil.rmtree(
                        settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True
                    )
                    # Deletion of the manual object will cascade in the DB
                    session.delete(manual_to_delete)
                    # We need to flush to ensure the delete is processed before 
                    # we add a newmanual with potentially the same unique constraints.
                    session.flush()
                else:
                    logger.info(f"New manual found: '{relative_path}'")

                if not pdf_data:
                    logger.warning(
                        f"Could not extract metadata from {relative_path}. Skipping."
                    )
                    continue

                new_manual = Manual(
                    file_name=Path(relative_path).name,
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

        # Handle deleted manuals
        deleted_paths = db_paths - fs_paths
        for path_to_delete in deleted_paths:
            manual_to_delete = db_manuals[path_to_delete]
            logger.info(f"'{path_to_delete}' has been removed. Deleting from database.")
            shutil.rmtree(
                settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True
            )
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
