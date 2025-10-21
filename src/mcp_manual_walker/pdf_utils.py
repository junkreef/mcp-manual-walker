import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


def scan_pdfs(root_dir: Path) -> Generator[Path, None, None]:
    """Recursively scans a directory for PDF files."""
    logger.info(f"Scanning for PDF files in {root_dir}...")
    for path in root_dir.rglob("*.pdf"):
        if path.is_file():
            yield path
    logger.info("PDF scan complete.")


def calculate_file_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_pdf_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extracts metadata and bookmarks from a PDF file.
    Returns a dictionary with document title and a list of bookmarks.
    """
    try:
        reader = PdfReader(str(file_path))

        # Extract document title
        document_title = reader.metadata.title if reader.metadata and reader.metadata.title else None

        # Extract bookmarks
        bookmarks = []
        if reader.outline:
            # PyPDF's outline is already hierarchical. We need to flatten it and add level info.
            def _traverse_bookmarks(outline, level=1):
                flat_list = []
                for item in outline:
                    if isinstance(item, list):
                        # This is a nested list of bookmarks
                        flat_list.extend(_traverse_bookmarks(item, level + 1))
                    else:
                        try:
                            # get_destination_page_number can raise an error if the destination is invalid
                            page_num = reader.get_destination_page_number(item)
                            if page_num is not None:
                                flat_list.append({
                                    "title": item.title,
                                    "level": level,
                                    "page_num": page_num + 1,  # pypdf is 0-indexed
                                })
                        except Exception as e:
                            logger.warning(f"Could not resolve page for bookmark '{item.title}' in {file_path}. Skipping. Error: {e}")

                return flat_list

            bookmarks = _traverse_bookmarks(reader.outline)

        return {
            "document_title": document_title,
            "bookmarks": bookmarks,
        }

    except PdfReadError as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path}: {e}")
        return None


def extract_text_from_page_range(file_path: Path, start_page: int, end_page: Optional[int] = None) -> str:
    """
    Extracts text from a given page range (inclusive) in a PDF file.
    Page numbers are 1-based.
    """
    if end_page is None or end_page < start_page:
        end_page = start_page

    text = ""
    try:
        reader = PdfReader(str(file_path))
        num_pages = len(reader.pages)

        # Adjust for 0-based indexing and validate page numbers
        start_idx = start_page - 1
        end_idx = end_page - 1

        if not (0 <= start_idx < num_pages):
            logger.error(f"Start page {start_page} is out of bounds for PDF with {num_pages} pages.")
            return ""

        # Ensure end page is within bounds
        end_idx = min(end_idx, num_pages - 1)

        for i in range(start_idx, end_idx + 1):
            page = reader.pages[i]
            text += page.extract_text() or ""

    except Exception as e:
        logger.error(f"Error extracting text from {file_path} for pages {start_page}-{end_page}: {e}")
        return ""

    return text


def create_temp_pdf_from_page_range(
    file_path: Path, start_page: int, end_page: Optional[int] = None
) -> Optional[Path]:
    """
    Creates a temporary PDF file from a given page range (inclusive).
    If end_page is None, it reads until the end of the document.
    Page numbers are 1-based.
    Returns the Path to the temporary file, or None on failure.
    """
    try:
        reader = PdfReader(str(file_path))
        writer = PdfWriter()
        num_pages = len(reader.pages)

        start_idx = start_page - 1

        # If end_page is not specified, use the last page of the document.
        if end_page is None:
            end_idx = num_pages - 1
        else:
            end_idx = end_page - 1

        # Validate start page
        if not (0 <= start_idx < num_pages):
            logger.error(
                f"Start page {start_page} is out of bounds for PDF with {num_pages} pages."
            )
            return None

        # Add pages to the writer, ensuring the end index is within bounds.
        end_idx = min(end_idx, num_pages - 1)
        for i in range(start_idx, end_idx + 1):
            writer.add_page(reader.pages[i])

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file_path = Path(temp_file.name)
        writer.write(temp_file)
        temp_file.close()  # Close the file handle

        logger.info(f"Created temporary PDF: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f"Failed to create temporary PDF from {file_path}: {e}")
        return None