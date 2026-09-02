import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Generator

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import NullObject

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
    Returns a dictionary with document title, a list of bookmarks, and page count.
    """
    try:
        reader = PdfReader(str(file_path))

        # Extract document title
        document_title = (
            reader.metadata.title if reader.metadata and reader.metadata.title else None
        )
        page_count = len(reader.pages)

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
                                # Safe extraction of Top
                                raw_top = getattr(item, "top", None)
                                top = None
                                if not isinstance(raw_top, (NullObject, type(None))):
                                    try:
                                        top = float(raw_top)
                                    except (ValueError, TypeError):
                                        logger.warning(
                                            f"Could not convert bookmark 'top' coordinate '{raw_top}' to float for bookmark '{item.title}'. Defaulting to None."
                                        )

                                flat_list.append(
                                    {
                                        "title": item.title,
                                        "level": level,
                                        "page_num": page_num + 1,  # pypdf is 0-indexed
                                        "top": top,  # Extract Y-coordinate safely
                                    }
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not resolve page for bookmark '{item.title}' in {file_path}. Skipping. Error: {e}"
                            )

                return flat_list

            bookmarks = _traverse_bookmarks(reader.outline)

        return {
            "document_title": document_title,
            "bookmarks": bookmarks,
            "page_count": page_count,
        }

    except PdfReadError as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path}: {e}")
        return None


def extract_pdf_fingerprint(pdf_path: Path) -> Dict[str, Any]:
    """
    Computes the file hash and extracts the pypdf metadata for a single PDF.

    This lives in pdf_utils rather than in the builder on purpose: it is
    submitted to a "spawn" process pool, and every worker imports the module
    that defines the task. Keeping it here means the workers only pull in pypdf
    instead of the heavy Docling/torch stack that builder.py imports.

    It never raises: any failure is reported back through the "error" key.
    """
    try:
        file_hash = calculate_file_hash(pdf_path)
        metadata = extract_pdf_metadata(pdf_path)
        if not metadata:
            return {
                "pdf_path": pdf_path,
                "file_hash": None,
                "metadata": None,
                "error": f"Failed to extract metadata from {pdf_path}",
            }
        return {
            "pdf_path": pdf_path,
            "file_hash": file_hash,
            "metadata": metadata,
            "error": None,
        }
    except Exception as e:
        return {
            "pdf_path": pdf_path,
            "file_hash": None,
            "metadata": None,
            "error": f"{type(e).__name__}: {e}",
        }
