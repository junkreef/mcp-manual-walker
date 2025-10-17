import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.mcp_manual_walker.models import Base, Bookmark, Cache, Manual


@pytest.fixture(scope="module")
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_manual_bookmark_cascade_delete(session):
    # Create a manual and bookmarks
    manual = Manual(
        file_name="test_manual.pdf",
        document_title="Test Manual",
        relative_path="test_manual.pdf",
        file_hash="123456",
    )
    session.add(manual)
    session.commit()

    bookmark1 = Bookmark(
        manual_id=manual.id,
        ordering=1,
        title="Chapter 1",
        level=1,
        page_num=1,
    )
    session.add(bookmark1)
    session.commit()

    # Verify everything was created
    assert session.query(Manual).count() == 1
    assert session.query(Bookmark).count() == 1

    # Delete the manual
    session.delete(manual)
    session.commit()

    # Verify bookmarks were deleted
    assert session.query(Manual).count() == 0
    assert session.query(Bookmark).count() == 0


def test_bookmark_cache_cascade_delete(session):
    # Create a manual, bookmark, and cache
    manual = Manual(
        file_name="test_manual_2.pdf",
        document_title="Test Manual 2",
        relative_path="test_manual_2.pdf",
        file_hash="abcdef",
    )
    session.add(manual)
    session.commit()

    bookmark = Bookmark(
        manual_id=manual.id,
        ordering=1,
        title="Chapter 1",
        level=1,
        page_num=1,
    )
    session.add(bookmark)
    session.commit()

    cache = Cache(
        bookmark_id=bookmark.id,
        manual_hash="abcdef",
        markdown_file_path="/tmp/cache.md",
    )
    session.add(cache)
    session.commit()

    # Verify everything was created
    assert session.query(Bookmark).count() == 1
    assert session.query(Cache).count() == 1

    # Delete the bookmark
    session.delete(bookmark)
    session.commit()

    # Verify cache was deleted
    assert session.query(Bookmark).count() == 0
    assert session.query(Cache).count() == 0