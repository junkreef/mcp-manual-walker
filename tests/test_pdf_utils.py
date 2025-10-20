import hashlib
from pathlib import Path

import pytest

from mcp_manual_walker.pdf_utils import (
    calculate_file_hash,
    extract_pdf_metadata,
)


@pytest.fixture(scope="session")
def pdf_for_utils_test(tmpdir_factory, dummy_pdf_factory):
    """
    Creates a specific dummy PDF for testing the PDF utils, with metadata and bookmarks.
    """
    pdf_path = Path(tmpdir_factory.mktemp("data").join("utils_test_manual.pdf"))
    dummy_pdf_factory(
        path=pdf_path,
        pages_content={1: "This is a test page."},
        metadata={"/Title": "Dummy Test Manual"},
        bookmarks={
            "Chapter 1: Introduction": (1, None),
            "Section 1.1: Overview": (1, "Chapter 1: Introduction"),
            "Section 1.2: Details": (1, "Chapter 1: Introduction"),
            "Chapter 2: Advanced Topics": (1, None),
        },
    )
    return pdf_path


def test_calculate_file_hash(pdf_for_utils_test: Path):
    """
    Tests that calculate_file_hash returns the correct SHA256 hash for a given file.
    """
    # Calculate hash using the utility function
    file_hash = calculate_file_hash(pdf_for_utils_test)

    # Calculate hash manually for verification
    hasher = hashlib.sha256()
    with open(pdf_for_utils_test, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    expected_hash = hasher.hexdigest()

    assert file_hash == expected_hash
    assert isinstance(file_hash, str)
    assert len(file_hash) == 64  # SHA256 hashes are 64 hex characters


def test_extract_pdf_metadata(pdf_for_utils_test: Path):
    """
    Tests the extraction of metadata and bookmarks from a PDF file.
    """
    pdf_data = extract_pdf_metadata(pdf_for_utils_test)

    # Verify document title
    assert pdf_data is not None
    assert pdf_data["document_title"] == "Dummy Test Manual"

    # Verify bookmarks structure and content
    bookmarks = pdf_data["bookmarks"]
    assert len(bookmarks) == 4

    # Check bookmark titles
    expected_titles = [
        "Chapter 1: Introduction",
        "Section 1.1: Overview",
        "Section 1.2: Details",
        "Chapter 2: Advanced Topics",
    ]
    actual_titles = [bm["title"] for bm in bookmarks]
    assert actual_titles == expected_titles

    # Check bookmark levels (hierarchy)
    # pypdf returns 1-based levels, so we expect [1, 2, 2, 1]
    expected_levels = [1, 2, 2, 1]
    actual_levels = [bm["level"] for bm in bookmarks]
    assert actual_levels == expected_levels

    # Check page numbers
    for bm in bookmarks:
        assert bm["page_num"] == 1  # pypdf returns 1-based page numbers


def test_extract_metadata_file_not_found():
    """
    Tests that extract_pdf_metadata returns None for a non-existent file.
    """
    non_existent_path = Path("non_existent_file.pdf")
    assert not non_existent_path.exists()
    result = extract_pdf_metadata(non_existent_path)
    assert result is None