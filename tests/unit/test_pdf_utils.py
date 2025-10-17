import hashlib
import os
from pathlib import Path
import pytest
from src.mcp_manual_walker.pdf_utils import (
    scan_pdfs,
    calculate_file_hash,
    extract_pdf_metadata,
    create_temp_pdf_from_page_range,
    extract_text_from_page_range,
)

@pytest.fixture(scope="module")
def assets_dir():
    return Path(__file__).parent.parent / "assets"

@pytest.fixture(scope="module")
def dummy_pdf(assets_dir):
    return assets_dir / "dummy.pdf"

def test_scan_pdfs(assets_dir):
    pdf_files = list(scan_pdfs(assets_dir))
    assert len(pdf_files) == 1
    assert pdf_files[0].name == "dummy.pdf"

def test_calculate_file_hash(dummy_pdf):
    hash_a = calculate_file_hash(dummy_pdf)
    hash_b = calculate_file_hash(dummy_pdf)
    assert hash_a == hash_b
    assert len(hash_a) == 64  # SHA256 length

    # Test that changing the file changes the hash
    with open(dummy_pdf, "ab") as f:
        f.write(b" ")
    hash_c = calculate_file_hash(dummy_pdf)
    assert hash_a != hash_c

    # Restore the file
    with open(dummy_pdf, "rb") as f:
        content = f.read()
    with open(dummy_pdf, "wb") as f:
        f.write(content[:-1])


def test_extract_pdf_metadata(dummy_pdf):
    metadata = extract_pdf_metadata(dummy_pdf)
    assert metadata is not None
    assert metadata["document_title"] == "Dummy PDF for Testing"
    assert len(metadata["bookmarks"]) == 2
    assert metadata["bookmarks"][0]["title"] == "Chapter 1"
    assert metadata["bookmarks"][0]["level"] == 1
    assert metadata["bookmarks"][1]["title"] == "Section 1.1"
    assert metadata["bookmarks"][1]["level"] == 2


def test_create_temp_pdf_from_page_range(dummy_pdf):
    # Test creating a temp PDF with a single page
    temp_pdf_path = create_temp_pdf_from_page_range(dummy_pdf, 1, 1)
    assert temp_pdf_path is not None
    assert temp_pdf_path.exists()
    os.remove(temp_pdf_path)

    # Test creating a temp PDF with multiple pages
    temp_pdf_path = create_temp_pdf_from_page_range(dummy_pdf, 1, 2)
    assert temp_pdf_path is not None
    assert temp_pdf_path.exists()
    os.remove(temp_pdf_path)

def test_extract_text_from_page_range_invalid_path():
    invalid_path = Path("non_existent_file.pdf")
    text = extract_text_from_page_range(invalid_path, 1, 1)
    assert text == ""