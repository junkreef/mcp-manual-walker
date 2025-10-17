import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from src.mcp_manual_walker.cache_utils import (
    create_cache,
    find_valid_cache,
    get_cache_filepath,
)
from src.mcp_manual_walker.models import Bookmark, Cache, Manual


@pytest.fixture
def mock_db_session():
    return MagicMock(spec=Session)


@pytest.fixture
def mock_bookmark():
    manual = Manual(
        id="manual1",
        file_name="test.pdf",
        file_hash="12345",
        relative_path="test.pdf",
    )
    bookmark = Bookmark(
        id="bookmark1", title="Test Bookmark", manual_id="manual1", manual=manual
    )
    bookmark.cache_entry = None
    return bookmark


def test_get_cache_filepath():
    filepath = get_cache_filepath("bookmark1", "12345")
    assert isinstance(filepath, Path)
    assert "12345" in filepath.name
    assert "bookmark1" in filepath.name


def test_find_valid_cache_hit(mock_db_session, mock_bookmark, tmp_path):
    # Setup cache entry and file
    mock_bookmark.manual.file_hash = "12345"
    cache_filepath = tmp_path / "cache.md"
    cache_filepath.write_text("cached content")

    mock_bookmark.cache_entry = Cache(
        manual_hash="12345", markdown_file_path=str(cache_filepath)
    )

    # Patch the Path object to use the temporary path
    with patch("src.mcp_manual_walker.cache_utils.Path", new=lambda x: Path(x)):
        content = find_valid_cache(mock_bookmark, mock_db_session)

    assert content == "cached content"


def test_find_valid_cache_miss_no_entry(mock_db_session, mock_bookmark):
    assert find_valid_cache(mock_bookmark, mock_db_session) is None


def test_find_valid_cache_miss_hash_mismatch(mock_db_session, mock_bookmark):
    mock_bookmark.manual.file_hash = "new_hash"
    mock_bookmark.cache_entry = Cache(manual_hash="old_hash")
    assert find_valid_cache(mock_bookmark, mock_db_session) is None


def test_find_valid_cache_miss_file_not_found(mock_db_session, mock_bookmark):
    mock_bookmark.manual.file_hash = "12345"
    mock_bookmark.cache_entry = Cache(
        manual_hash="12345", markdown_file_path="nonexistent/file.md"
    )
    with patch("src.mcp_manual_walker.cache_utils.Path.is_file", return_value=False):
        assert find_valid_cache(mock_bookmark, mock_db_session) is None
    mock_db_session.delete.assert_called_once_with(mock_bookmark.cache_entry)


def test_create_cache(mock_db_session, mock_bookmark, tmp_path):
    content = "new cached content"
    mock_bookmark.manual.file_hash = "12345"

    with patch("src.mcp_manual_walker.cache_utils.get_cache_filepath") as mock_get_path:
        cache_filepath = tmp_path / "new_cache.md"
        mock_get_path.return_value = cache_filepath

        create_cache(mock_bookmark, content, mock_db_session)

    assert cache_filepath.read_text() == content
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()