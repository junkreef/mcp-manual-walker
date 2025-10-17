import os
import time
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mcp_manual_walker.cache_utils import create_cache, find_valid_cache
from mcp_manual_walker.models import Base, Bookmark, Cache, Manual


@pytest.fixture(scope="function")
def db_session():
    """
    Fixture to set up an in-memory SQLite database for each test function.
    This ensures that tests are isolated and don't interfere with each other.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def test_data(db_session):
    """
    Populates the database with a manual and a bookmark for testing.
    """
    manual = Manual(
        id="manual-1",
        file_name="test.pdf",
        document_title="Test Manual",
        relative_path="test.pdf",
        file_hash="hash-123",
    )
    bookmark = Bookmark(
        id="bookmark-1",
        manual_id=manual.id,
        ordering=1,
        title="Chapter 1",
        level=1,
        page_num=1,
    )
    db_session.add(manual)
    db_session.add(bookmark)
    db_session.commit()
    return manual, bookmark


def test_cascade_delete(db_session, test_data):
    """
    Tests that deleting a Manual also deletes its associated Bookmarks and Caches.
    This verifies the `cascade="all, delete-orphan"` setting in the models.
    """
    manual, bookmark = test_data
    manual_id = manual.id
    bookmark_id = bookmark.id

    # Create a dummy cache entry to test cascade deletion
    cache_path = Path("./test_cache.md")
    with open(cache_path, "w") as f:
        f.write("cached content")

    cache_entry = Cache(
        id="cache-1",
        bookmark_id=bookmark_id,
        manual_hash=manual.file_hash,
        markdown_file_path=str(cache_path),
    )
    db_session.add(cache_entry)
    db_session.commit()

    # Ensure everything is in the database before deletion
    assert db_session.query(Manual).filter_by(id=manual_id).first() is not None
    assert db_session.query(Bookmark).filter_by(id=bookmark_id).first() is not None
    assert db_session.query(Cache).filter_by(id="cache-1").first() is not None

    # Delete the manual
    db_session.delete(manual)
    db_session.commit()

    # Verify that the manual and all its children are deleted
    assert db_session.query(Manual).filter_by(id=manual_id).first() is None
    assert db_session.query(Bookmark).filter_by(id=bookmark_id).first() is None
    assert db_session.query(Cache).filter_by(id="cache-1").first() is None

    # Clean up the dummy cache file
    os.remove(cache_path)


def test_find_valid_cache_hit(db_session, test_data, tmp_path):
    """
    Tests a cache hit scenario where a valid cache entry exists.
    """
    manual, bookmark = test_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / f"{uuid4()}.md"
    expected_content = "This is cached content."
    cache_file.write_text(expected_content)

    cache_entry = Cache(
        bookmark_id=bookmark.id,
        manual_hash=manual.file_hash,
        markdown_file_path=str(cache_file),
    )
    bookmark.cache_entry = cache_entry  # Associate cache with bookmark
    db_session.add(cache_entry)
    db_session.commit()

    # The bookmark object needs to be refreshed to load the new relationship
    db_session.refresh(bookmark)

    # find_valid_cache should return the content
    content = find_valid_cache(bookmark, db_session)
    assert content == expected_content


def test_find_valid_cache_miss_hash_mismatch(db_session, test_data, tmp_path):
    """
    Tests a cache miss scenario where the manual's hash has changed.
    """
    manual, bookmark = test_data
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / f"{uuid4()}.md"
    cache_file.write_text("outdated content")

    cache_entry = Cache(
        bookmark_id=bookmark.id,
        manual_hash="old-hash-456",  # Different from the manual's current hash
        markdown_file_path=str(cache_file),
    )
    bookmark.cache_entry = cache_entry
    db_session.add(cache_entry)
    db_session.commit()
    db_session.refresh(bookmark)

    # find_valid_cache should return None because the hash is invalid
    content = find_valid_cache(bookmark, db_session)
    assert content is None


def test_find_valid_cache_miss_no_entry(db_session, test_data):
    """
    Tests a cache miss scenario where no cache entry exists for the bookmark.
    """
    manual, bookmark = test_data
    content = find_valid_cache(bookmark, db_session)
    assert content is None


def test_create_cache(db_session, test_data, tmp_path):
    """
    Tests the creation of a new cache entry.
    """
    manual, bookmark = test_data
    content_to_cache = "Newly generated markdown content."

    # Mock the settings to use the temporary directory
    from mcp_manual_walker import config
    original_cache_dir = config.settings.CACHE_DIR
    config.settings.CACHE_DIR = tmp_path

    try:
        # Create the cache
        create_cache(bookmark, content_to_cache, db_session)

        # Verify the cache entry in the database
        cache_entry = db_session.query(Cache).filter_by(bookmark_id=bookmark.id).one()
        assert cache_entry is not None
        assert cache_entry.manual_hash == manual.file_hash
        assert Path(cache_entry.markdown_file_path).name.endswith(".md")

        # Verify the content of the cache file
        with open(cache_entry.markdown_file_path, "r") as f:
            assert f.read() == content_to_cache
    finally:
        # Restore the original settings
        config.settings.CACHE_DIR = original_cache_dir