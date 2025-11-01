import shutil
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal, init_db
from .models import Bookmark, Manual
from .pdf_utils import (
    calculate_file_hash,
    extract_pdf_metadata,
    scan_pdfs,
)


def sync_database():
    """
    Scans the PDF directory and synchronizes the database.
    - Adds new manuals.
    - Updates manuals that have changed.
    - Deletes manuals that are no longer on the filesystem.
    """
    print("Starting database synchronization...")

    # Ensure DB and tables exist
    init_db()

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
                print(f"ERROR: Could not calculate hash for {pdf_path}: {e}")
                continue

            if (
                relative_path not in db_paths
                or db_manuals[relative_path].file_hash != file_hash
            ):
                manual_to_delete = db_manuals.get(relative_path)
                if manual_to_delete:
                    print(f"INFO: '{relative_path}' has been updated. Re-processing.")
                    shutil.rmtree(
                        settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True
                    )
                    db.delete(manual_to_delete)
                    db.commit()
                else:
                    print(f"INFO: New manual found: '{relative_path}'")

                pdf_data = extract_pdf_metadata(pdf_path)
                if not pdf_data:
                    print(
                        f"WARNING: Could not extract metadata from {pdf_path}. Skipping."
                    )
                    continue

                new_manual = Manual(
                    file_name=pdf_path.name,
                    document_title=pdf_data["document_title"],
                    relative_path=relative_path,
                    file_hash=file_hash,
                    page_count=pdf_data["page_count"],
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
                    db.flush()
                    parent_stack[level] = bookmark
                    keys_to_del = [k for k in parent_stack if k > level]
                    for k in keys_to_del:
                        del parent_stack[k]
                print(f"INFO: Successfully processed and added '{relative_path}'.")

        deleted_paths = db_paths - fs_paths
        for path_to_delete in deleted_paths:
            manual_to_delete = db_manuals[path_to_delete]
            print(f"INFO: '{path_to_delete}' has been removed. Deleting from database.")
            shutil.rmtree(settings.CACHE_DIR / manual_to_delete.id, ignore_errors=True)
            db.delete(manual_to_delete)

        db.commit()

    except Exception as e:
        print(f"ERROR: An error occurred during database synchronization: {e}")
        db.rollback()
    finally:
        db.close()
    print("Database synchronization complete.")


if __name__ == "__main__":
    sync_database()
