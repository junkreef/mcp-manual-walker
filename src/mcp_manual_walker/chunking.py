import logging
from typing import Any, Dict, List, Optional

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from mcp_manual_walker.config import settings
from mcp_manual_walker.models import Bookmark, Manual

logger = logging.getLogger(__name__)


def chunk_text_by_coordinates(doc, manual: Manual) -> List[Dict[str, Any]]:
    """
    Chunks document text by mapping Docling text items to existing DB bookmarks
    using page number and Y-coordinates (Top).
    """
    chunks = []

    # Check if doc has texts
    if not hasattr(doc, "texts"):
        text = doc.export_to_markdown()
        return [
            {"text": text, "metadata": {"manual_id": manual.id, "bookmark_id": None}}
        ]

    # 1. Organize Bookmarks
    # manual.bookmarks should be available (lazy loaded by session)
    # Filter out bookmarks without page_num (should n't happen if inserted correctly)
    raw_bookmarks = [bm for bm in manual.bookmarks]

    # Pre-process: If some bookmarks lack coordinates, try to find them in the text
    # This is a heuristic fallback.
    
    # Optimization: Group texts by page first to avoid O(N*M) complexity
    texts_by_page: Dict[int, List[Any]] = {}
    for item in doc.texts:
        if item.prov:
            prov = item.prov[0]
            page_no = prov.page_no
            if page_no not in texts_by_page:
                texts_by_page[page_no] = []
            texts_by_page[page_no].append(item)

    for bm in raw_bookmarks:
        if bm.page_top is None:
            # Try to find the first occurrence of the title on the assigned page
            # Both bm.page_num and docling page_no are 1-indexed
            page_texts = texts_by_page.get(bm.page_num, [])
            for item in page_texts:
                if bm.title.lower() in item.text.lower():
                    # We know prov exists from the grouping logic
                    prov = item.prov[0]
                    bm.page_top = getattr(prov.bbox, "t", 0.0)
                    logger.info(f"Resolved missing coordinate for bookmark '{bm.title}' on page {bm.page_num} to {bm.page_top}")
                    break

    bms_by_page: Dict[int, List[Bookmark]] = {}
    for bm in raw_bookmarks:
        if bm.page_num not in bms_by_page:
            bms_by_page[bm.page_num] = []
        bms_by_page[bm.page_num].append(bm)

    # Sort each page's bookmarks by Top DESC for the scanning logic
    for p in bms_by_page:
        bms_by_page[p].sort(
            key=lambda x: x.page_top if x.page_top is not None else -1.0, reverse=True
        )

    current_bookmark: Optional[Bookmark] = None
    current_text_buffer: List[str] = []



    # Initialize Text Splitter for large chunks
    # We use Markdown splitter to respect table structure etc.
    # Overlap is enabled to avoid context loss.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    def add_chunk(content: str, bookmark: Optional[Bookmark]):
        if not content:
            return
        
        # Split large content
        sub_chunks = splitter.split_text(content)
        
        for sub_chunk in sub_chunks:
            chunks.append(
                {
                    "text": sub_chunk,
                    "metadata": {
                        "manual_id": manual.id,
                        "bookmark_id": bookmark.id if bookmark else None,
                    },
                }
            )

    for item in doc.texts:
        text = item.text.strip()
        if not text:
            continue

        if not item.prov:
            # Fallback: keep current bookmark
            current_text_buffer.append(text)
            continue

        # Use first provenance item for location
        prov = item.prov[0]
        page_no = prov.page_no

        # Docling bbox is usually [l, b, r, t] (bottom-left origin)
        # We need Top Y.
        # docling_core types usually have .l, .r, .t, .b properties.
        item_top = getattr(prov.bbox, "t", 0.0)

        # Match Bookmark
        page_bms = bms_by_page.get(page_no, [])

        candidate = None
        # Iterate bookmarks on this page (Sorted Top DESC)
        for bm in page_bms:
            bm_top = bm.page_top if bm.page_top is not None else -1.0

            # Fuzzy Logic:
            # If item is below bookmark (Item Top < BM Top)
            if item_top < (bm_top + 10.0):
                candidate = bm
            else:
                break

        if candidate:
            if current_bookmark != candidate:
                # Context Switch!
                # Flush buffer
                if current_text_buffer:
                    chunk_content = "\n\n".join(current_text_buffer)
                    add_chunk(chunk_content, current_bookmark)
                    current_text_buffer = []
                current_bookmark = candidate

        current_text_buffer.append(text)

    # Flush final buffer
    if current_text_buffer:
        chunk_content = "\n\n".join(current_text_buffer)
        add_chunk(chunk_content, current_bookmark)

    return chunks
