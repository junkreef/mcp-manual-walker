import logging
from pathlib import Path
from typing import Optional
from .config import settings
from .models import Cache, Bookmark, Manual
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

def get_cache_filepath(bookmark_id: str, manual_hash: str) -> Path:
    """Generates a consistent filepath for a cached bookmark."""
    return settings.CACHE_DIR / f"{manual_hash[:16]}_{bookmark_id}.md"

def find_valid_cache(bookmark: Bookmark, db: Session) -> Optional[str]:
    """
    Checks for a valid cache entry for a bookmark.
    Returns the markdown content if a valid cache file exists.
    """
    if bookmark.cache_entry and bookmark.cache_entry.manual_hash == bookmark.manual.file_hash:
        cache_path = Path(bookmark.cache_entry.markdown_file_path)
        if cache_path.is_file():
            logger.info(f"Valid cache found for bookmark '{bookmark.title}' in '{bookmark.manual.file_name}'.")
            return cache_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Cache entry exists for '{bookmark.title}' but file is missing. Will regenerate.")
            # The cache entry is stale, delete it.
            db.delete(bookmark.cache_entry)
            db.commit()
    return None

def create_cache(bookmark: Bookmark, content: str, db: Session) -> None:
    """Creates a new cache entry and saves the content to a file."""
    manual = bookmark.manual
    cache_filepath = get_cache_filepath(bookmark.id, manual.file_hash)

    try:
        # Write the content to the cache file
        cache_filepath.write_text(content, encoding="utf-8")

        # If there's an old cache entry, delete it first
        if bookmark.cache_entry:
            db.delete(bookmark.cache_entry)
            db.flush() # Use flush to ensure the delete is processed before the insert

        # Create a new cache record in the database
        new_cache = Cache(
            bookmark_id=bookmark.id,
            manual_hash=manual.file_hash,
            markdown_file_path=str(cache_filepath)
        )
        db.add(new_cache)
        db.commit()
        logger.info(f"Cache created for bookmark '{bookmark.title}' at '{cache_filepath}'.")

    except IOError as e:
        logger.error(f"Failed to write cache file for bookmark '{bookmark.title}': {e}")
        db.rollback()
    except Exception as e:
        logger.error(f"An unexpected error occurred during cache creation for '{bookmark.title}': {e}")
        db.rollback()