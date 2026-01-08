import logging
from typing import Any, Dict, List, Optional

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
            # bm.page_num is 1-indexed, docling page_no is 0-indexed
            page_texts = texts_by_page.get(bm.page_num - 1, [])
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
        # Check prov.bbox structure. In inspect_prov_bbox.py output:
        # BBox: l=186.0 t=291.92 r=275.82 b=257.0
        # It has .t attribute? Or is it a dict?
        # Output was `BBox: ... t=...` via string representation.
        # docling_core types usually have .l, .r, .t, .b properties.
        # Let's assume .t exists.

        item_top = getattr(prov.bbox, "t", 0.0)

        # Match Bookmark
        page_bms = bms_by_page.get(page_no, [])

        candidate = None
        # Iterate bookmarks on this page (Sorted Top DESC)
        for bm in page_bms:
            bm_top = bm.page_top if bm.page_top is not None else -1.0

            # Fuzzy Logic:
            # If bookmark represents a header, the header text itself is at 'bm_top'.
            # Content follows below.
            # So item belongs to bookmark if item is BELOW the bookmark (Item Top < BM Top).
            # Wait, Y-axis increases Upwards (Bottom-Left origin).
            # So Higher Value = Higher on Page.
            # So "Below" means Item Top < BM Top.
            # Tolerance: If item is strictly equal or slightly above due to floats?
            # Epsilon = 10.0 units (~3-4mm)

            if item_top < (bm_top + 10.0):
                # Valid candidate (Item is below Bookmark)
                # Since we iterate from Highest BM to Lowest BM,
                # the *first* BM we encounter that satisfies this is the Highest one.
                # But we want the *lowest* BM that is still above the item (Immediate Parent).
                # Example:
                # BM A (700)
                # BM B (600)
                # Item (550) -> 550 < 700 (True), 550 < 600 (True).
                # We want BM B.
                # So we want the LAST candidate in this sorted list.
                # Wait, if we iterate DESC (700, 600)
                # 700 matches. 600 matches.
                # The last matching one is 600.
                candidate = bm
            else:
                # Item is ABOVE this bookmark (Item 650 > BM 600).
                # Stops matching further lower bookmarks.
                break

        # If we found a new candidate on this page (or confirmed one), update?
        # Logic: A page flow updates the context.
        # Only update if candidate is found?
        # If no candidate found (e.g. item is at very top of page 800, first BM starts at 500),
        # then it belongs to previous page's context.
        # EXCEPT if there are bookmarks on this page, and we are above all of them?
        # Yes, implies continuation.

        if candidate:
            if current_bookmark != candidate:
                # Context Switch!
                # Flush buffer
                if current_text_buffer:
                    chunk_content = "\n\n".join(current_text_buffer)
                    chunks.append(
                        {
                            "text": chunk_content,
                            "metadata": {
                                "manual_id": manual.id,
                                "bookmark_id": current_bookmark.id
                                if current_bookmark
                                else None,
                            },
                        }
                    )
                    current_text_buffer = []
                current_bookmark = candidate

        current_text_buffer.append(text)

    # Flush final buffer
    if current_text_buffer:
        chunk_content = "\n\n".join(current_text_buffer)
        chunks.append(
            {
                "text": chunk_content,
                "metadata": {
                    "manual_id": manual.id,
                    "bookmark_id": current_bookmark.id if current_bookmark else None,
                },
            }
        )

    return chunks
