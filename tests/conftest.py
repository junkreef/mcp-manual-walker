from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@pytest.fixture(scope="session")
def dummy_pdf_factory():
    """
    A factory fixture to create dummy PDF files with hierarchical bookmarks for testing.
    """
    def _create_pdf(
        path: Path,
        pages_content: dict[int, str],
        bookmarks: dict[str, tuple[int, str | None]] | None = None,
        metadata: dict[str, str] | None = None,
    ):
        """
        Creates a dummy PDF file.

        Args:
            path: The path to save the PDF file.
            pages_content: A dictionary mapping page numbers (1-based) to content
                strings.
            bookmarks: A dictionary for hierarchical bookmarks.
                Format: {
                    "Bookmark Title": (page_number, parent_title | None),
                    ...
                }
            metadata: A dictionary for the document's metadata
                (e.g., {"/Title": "My Title"}).
        """
        # Create the PDF with page content using reportlab
        c = canvas.Canvas(str(path), pagesize=letter)
        num_pages = max(pages_content.keys()) if pages_content else 1
        for i in range(1, num_pages + 1):
            content = pages_content.get(i, f"This is default content for page {i}.")
            c.drawString(100, 750, content)
            c.showPage()
        c.save()

        # Re-open with pypdf to add metadata and bookmarks
        reader = PdfReader(str(path))
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)

        if metadata:
            writer.add_metadata(metadata)

        if bookmarks:
            outline_map = {}
            # First pass for top-level bookmarks
            for title, (page_num, parent_title) in bookmarks.items():
                if parent_title is None:
                    parent_item = writer.add_outline_item(title, page_num - 1)
                    outline_map[title] = parent_item

            # Second pass for child bookmarks
            for title, (page_num, parent_title) in bookmarks.items():
                if parent_title is not None and parent_title in outline_map:
                    parent_item = outline_map[parent_title]
                    child_item = writer.add_outline_item(
                        title, page_num - 1, parent=parent_item
                    )
                    outline_map[title] = child_item # Allow for deeper nesting

        with open(path, "wb") as f:
            writer.write(f)

        return path

    return _create_pdf
