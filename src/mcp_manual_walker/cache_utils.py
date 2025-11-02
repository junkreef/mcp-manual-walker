from datetime import datetime, UTC
import logging
import os
from pathlib import Path
from typing import List, Optional

from sqlalchemy import update
from sqlalchemy.orm import Session

from .config import settings
from .models import Cache, Manual

logger = logging.getLogger(__name__)


def get_cache_filepath(manual_id: str, page_num: int) -> Path:
    """
    Generates a consistent, structured filepath for a cached page.
    Ensures the subdirectory for the manual exists.
    e.g., cache/MANUAL_ID/page_PAGE_NUM.md
    """
    manual_cache_dir = settings.CACHE_DIR / manual_id
    if manual_cache_dir.is_file():
        logger.critical(
            f"Cache path {manual_cache_dir} exists as a file, not a directory. Manual cache is corrupted."
        )
        raise FileExistsError(f"Cache path {manual_cache_dir} is a file.")
    manual_cache_dir.mkdir(parents=True, exist_ok=True)
    return manual_cache_dir / f"page_{page_num}.md"


def find_page_cache(manual: Manual, page_num: int, db: Session) -> Optional[str]:
    """
    Checks for a valid cache entry for a single page of a manual.
    Returns the markdown content if a valid cache file exists.
    """
    cache_entry = db.query(Cache).filter_by(manual_id=manual.id, page_num=page_num).first()

    if cache_entry and cache_entry.manual_hash == manual.file_hash:
        cache_path = get_cache_filepath(manual.id, page_num)
        if cache_path.is_file():
            logger.debug(f"Valid cache found for page {page_num} in '{manual.file_name}'.")
            return cache_path.read_text(encoding="utf-8")
        else:
            logger.warning(
                f"Cache entry for page {page_num} of '{manual.file_name}' exists in DB but file is missing. Deleting entry."
            )
            db.delete(cache_entry)
    return None


def create_page_cache(manual: Manual, page_num: int, content: str, db: Session) -> None:
    """
    Creates a new cache entry for a single page and saves the content to a file.
    Does not commit the transaction; caller is responsible.
    """
    cache_filepath = get_cache_filepath(manual.id, page_num)
    now = datetime.now(UTC)

    # Write the content to the cache file
    cache_filepath.write_text(content, encoding="utf-8")

    # Use merge to either insert a new record or update an existing one.
    # This handles cases where a stale record might exist.
    new_cache = Cache(
        manual_id=manual.id,
        page_num=page_num,
        manual_hash=manual.file_hash,
        created_at=now,
        last_accessed_at=now,
    )
    db.merge(new_cache)
    logger.debug(f"Cache created for page {page_num} of '{manual.file_name}' at '{cache_filepath}'.")


def batch_update_last_accessed(manual_id: str, page_nums: List[int], db: Session) -> None:
    """
    Updates the last_accessed_at timestamp for a list of pages in a single batch operation.
    Does not commit the transaction; caller is responsible.
    """
    if not page_nums:
        return

    stmt = (
        update(Cache)
        .where(Cache.manual_id == manual_id)
        .where(Cache.page_num.in_(page_nums))
        .values(last_accessed_at=datetime.now(UTC))
    )
    db.execute(stmt)
    logger.debug(f"Updated last_accessed_at for {len(page_nums)} pages in manual {manual_id}.")