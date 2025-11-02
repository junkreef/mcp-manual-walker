import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mcp_manual_walker import config
from mcp_manual_walker.cache_utils import (
    batch_update_last_accessed,
    create_page_cache,
    find_page_cache,
    get_cache_filepath,
)
from mcp_manual_walker.models import Base, Bookmark, Cache, Manual


@pytest.fixture(scope="function")
def db_session():
    """
    Fixture to set up an in-memory SQLite database for each test function.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def test_manual(db_session):
    """
    Populates the database with a manual for testing.
    """
    manual = Manual(
        id="manual-1",
        file_name="test.pdf",
        document_title="Test Manual",
        relative_path="test.pdf",
        file_hash="hash-123",
        page_count=10,
    )
    db_session.add(manual)
    db_session.commit()
    db_session.refresh(manual)
    return manual


def test_cascade_delete(db_session, test_manual):
    """
    Tests that deleting a Manual also deletes its associated Cache entries.
    """
    manual = test_manual
    page_to_cache = 1

    # Create a dummy cache entry
    cache_entry = Cache(
        manual_id=manual.id,
        page_num=page_to_cache,
        manual_hash=manual.file_hash,
    )
    db_session.add(cache_entry)
    db_session.commit()

    assert db_session.query(Cache).filter_by(manual_id=manual.id).first() is not None

    # Delete the manual
    db_session.delete(manual)
    db_session.commit()

    # Verify that the cache entry is also deleted
    assert db_session.query(Cache).filter_by(manual_id=manual.id).first() is None


def test_find_page_cache_hit(db_session, test_manual, tmp_path, monkeypatch):
    """
    Tests a cache hit scenario for find_page_cache.
    """
    monkeypatch.setattr(config.settings, "CACHE_DIR", tmp_path)
    manual = test_manual
    page_num = 3
    expected_content = "This is cached content for page 3."

    # Manually create cache file and DB entry
    cache_path = get_cache_filepath(manual.id, page_num)
    cache_path.write_text(expected_content)
    db_session.add(Cache(manual_id=manual.id, page_num=page_num, manual_hash=manual.file_hash))
    db_session.commit()

    content = find_page_cache(manual, page_num, db_session)
    assert content == expected_content


def test_find_page_cache_miss_hash_mismatch(db_session, test_manual, tmp_path, monkeypatch):
    """
    Tests a cache miss due to a hash mismatch.
    """
    monkeypatch.setattr(config.settings, "CACHE_DIR", tmp_path)
    manual = test_manual
    page_num = 4

    # Create a cache entry with an old hash
    db_session.add(Cache(manual_id=manual.id, page_num=page_num, manual_hash="old-hash-456"))
    db_session.commit()

    content = find_page_cache(manual, page_num, db_session)
    assert content is None


def test_create_page_cache(db_session, test_manual, tmp_path, monkeypatch):
    """
    Tests the creation of a new page cache entry.
    """
    monkeypatch.setattr(config.settings, "CACHE_DIR", tmp_path)
    manual = test_manual
    page_num = 5
    content_to_cache = "Content for page 5."

    create_page_cache(manual, page_num, content_to_cache, db_session)

    # Verify DB entry
    cache_entry = db_session.query(Cache).filter_by(manual_id=manual.id, page_num=page_num).one()
    assert cache_entry is not None
    assert cache_entry.manual_hash == manual.file_hash

    # Verify file content
    cache_path = get_cache_filepath(manual.id, page_num)
    assert cache_path.read_text() == content_to_cache


def test_batch_update_last_accessed(db_session, test_manual):
    """
    Tests that last_accessed_at is updated in a batch.
    """
    manual = test_manual
    pages_to_update = [1, 2, 3]

    # Create initial entries
    for page_num in pages_to_update:
        db_session.add(Cache(manual_id=manual.id, page_num=page_num, manual_hash=manual.file_hash))
    db_session.commit()

    # Get initial timestamps
    initial_entries = db_session.query(Cache).filter(Cache.page_num.in_(pages_to_update)).all()
    initial_timestamp = initial_entries[0].last_accessed_at

    # Wait a bit to ensure the new timestamp is different
    time.sleep(0.01)

    # Run batch update
    batch_update_last_accessed(manual.id, pages_to_update, db_session)
    db_session.commit()

    # Verify timestamps are updated
    updated_entries = db_session.query(Cache).filter(Cache.page_num.in_(pages_to_update)).all()
    for entry in updated_entries:
        assert entry.last_accessed_at > initial_timestamp