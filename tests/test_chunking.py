from unittest.mock import MagicMock

import pytest

from mcp_manual_walker.chunking import chunk_text_by_coordinates
from mcp_manual_walker.models import Bookmark, Manual


@pytest.fixture
def mock_manual():
    manual = MagicMock(spec=Manual)
    manual.id = "test_manual_id"
    manual.bookmarks = []
    # Configure bookmarks (lazy loaded usually, but mock list)
    return manual


def test_chunking_fallback_no_prov(mock_manual):
    """Test chunking when docling items have no provenance (fallback)."""
    # Create doc mock
    doc = MagicMock()
    item1 = MagicMock()
    item1.text = "Start Text"
    item1.prov = []  # No prov

    item2 = MagicMock()
    item2.text = "More Text"
    item2.prov = []

    doc.texts = [item1, item2]

    chunks = chunk_text_by_coordinates(doc, mock_manual)

    assert len(chunks) == 1
    assert "Start Text" in chunks[0]["text"]
    assert "More Text" in chunks[0]["text"]
    assert chunks[0]["metadata"]["manual_id"] == "test_manual_id"
    assert chunks[0]["metadata"]["bookmark_id"] is None


def test_chunking_with_bookmarks(mock_manual):
    """Test chunking with bookmarks and provenance."""
    # Setup Bookmarks
    # Page 1, Top 800 (Header 1)
    # Page 1, Top 600 (Header 2)
    bm1 = MagicMock(spec=Bookmark)
    bm1.id = "bm1"
    bm1.page_num = 1
    bm1.page_top = 800.0
    bm1.title = "Header 1"

    bm2 = MagicMock(spec=Bookmark)
    bm2.id = "bm2"
    bm2.page_num = 1
    bm2.page_top = 600.0
    bm2.title = "Header 2"

    # Needs to be iterable
    mock_manual.bookmarks = [bm1, bm2]

    # Setup Doc Items
    doc = MagicMock()

    # Item 1: Top 750 (Below Header 1 (800), Above Header 2 (600)) -> bm1
    item1 = MagicMock()
    item1.text = "Content for Header 1"
    item1.prov = [MagicMock(page_no=1, bbox=MagicMock(t=750.0))]

    # Item 2: Top 550 (Below Header 2 (600)) -> bm2
    item2 = MagicMock()
    item2.text = "Content for Header 2"
    item2.prov = [MagicMock(page_no=1, bbox=MagicMock(t=550.0))]

    # Item 3: Page 2, Top 800 (No BMs on page 2) -> Should continue bm2 (current)
    # Or reset?
    # Logic: if no candidate found on page, context persists?
    # Logic says: if bms_by_page.get(page_no) is empty, candidate=None.
    # if candidate: update current.
    # So if candidate is None, current_bookmark stays same.
    item3 = MagicMock()
    item3.text = "Content on Page 2"
    item3.prov = [MagicMock(page_no=2, bbox=MagicMock(t=800.0))]

    doc.texts = [item1, item2, item3]

    chunks = chunk_text_by_coordinates(doc, mock_manual)

    # Expectation:
    # 1. First text -> Matches bm1.
    # Buffer: ["Content for Header 1"] (current=bm1)
    # 2. Second text -> Matches bm2. Context switch!
    # Chunk 1 emitted (bm1, "Content for Header 1")
    # Buffer: ["Content for Header 2"] (current=bm2)
    # 3. Third text -> Page 2. No BMs. Candidate=None.
    # Context stays bm2.
    # Buffer: ["Content for Header 2", "Content on Page 2"]
    # End -> Flush Chunk 2 (bm2, "Content for Header 2\n\nContent on Page 2")

    assert len(chunks) == 2

    # Chunk 1
    assert chunks[0]["metadata"]["bookmark_id"] == "bm1"
    assert "Content for Header 1" in chunks[0]["text"]

    # Chunk 2
    assert chunks[1]["metadata"]["bookmark_id"] == "bm2"
    assert "Content for Header 2" in chunks[1]["text"]
    assert "Content on Page 2" in chunks[1]["text"]


def test_chunking_no_doc_texts(mock_manual):
    """Test fallback when doc has no texts attribute (e.g. Scanned PDF fallback not fully supported)."""
    doc = MagicMock()
    del doc.texts  # Ensure no texts attr
    doc.export_to_markdown.return_value = "Full Generic Text"

    chunks = chunk_text_by_coordinates(doc, mock_manual)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Full Generic Text"
    assert chunks[0]["metadata"]["bookmark_id"] is None

def test_chunking_large_content(mock_manual):
    """Test splitting of large content using LangChain splitter."""
    doc = MagicMock()
    
    # Create a large text (> 2000 chars)
    # 2500 chars
    large_text = "A" * 1500 + "\n\n" + "B" * 1000
    
    item1 = MagicMock()
    item1.text = large_text
    item1.prov = [] # Fallback logic uses append then flush called add_chunk
    
    doc.texts = [item1]
    
    chunks = chunk_text_by_coordinates(doc, mock_manual)
    
    # Expectation:
    # Total 2500+ chars. Max chunk 2000.
    # Should be at least 2 chunks.
    # "A"*1500 + \n\n is considered a separator. 
    # LangChain should split intelligently. 
    # Likely ["A"*1500, "B"*1000] if separator works well.
    
    assert len(chunks) == 2
    
    # Concatenation should restore full text (minus potential whitespace variance if splitter strips)
    # But RecursiveCharacterTextSplitter with overlap=0 should preserve content if just split.
    # Actually splitter might consume separators.
    
    full_text_out = "".join([c["text"] for c in chunks])
    # check that we have roughly the content
    assert "AAAA" in full_text_out
    assert "BBBB" in full_text_out
    
    # Check Metadata
    assert chunks[0]["metadata"]["manual_id"] == "test_manual_id"
    assert chunks[1]["metadata"]["manual_id"] == "test_manual_id"


def test_chunking_sequence_and_overlap(mock_manual):
    """Test chunking of continuous text to ensure correct overlap creation."""
    doc = MagicMock()
    
    # Create text that will force a split in the middle of a line if no separators found?
    # Or provide clear separators.
    # Create continuous text to force a split in the middle, ensuring overlap
    full = "P" * 2500
    
    item = MagicMock()
    item.text = full
    item.prov = []
    
    doc.texts = [item]
    
    chunks = chunk_text_by_coordinates(doc, mock_manual)
    
    assert len(chunks) == 2
    
    # Chunk 0 should be 2000 chars
    assert len(chunks[0]["text"]) == 2000
    
    # Chunk 1 should be 500 chars (remainder) + 200 overlap = 700 chars
    assert len(chunks[1]["text"]) == 700
    
    # Verify overlap
    chunk0_suffix = chunks[0]["text"][-200:]
    chunk1_prefix = chunks[1]["text"][:200]
    assert chunk0_suffix == chunk1_prefix
    
    # Verify metadata is identical
    assert chunks[0]["metadata"] == chunks[1]["metadata"]


def test_chunking_fallback_resolves_missing_bookmark_coordinate(mock_manual):
    """Test that a bookmark without page_top is resolved via title search on its
    own (1-indexed) page, not the previous page."""
    bm = MagicMock(spec=Bookmark)
    bm.id = "bm_troubleshooting"
    bm.page_num = 3
    bm.page_top = None
    bm.title = "Troubleshooting"

    mock_manual.bookmarks = [bm]

    doc = MagicMock()

    # Page 2 item: same top as the page-3 title, but doesn't contain it.
    # Presence of this item on the wrong page guards against the old
    # off-by-one bug (page_num - 1) accidentally matching.
    item_page2 = MagicMock()
    item_page2.text = "Unrelated heading"
    item_page2.prov = [MagicMock(page_no=2, bbox=MagicMock(t=750.0))]

    # Page 3 item A: the bookmark title itself, used to resolve page_top.
    item_page3_title = MagicMock()
    item_page3_title.text = "3 Troubleshooting"
    item_page3_title.prov = [MagicMock(page_no=3, bbox=MagicMock(t=750.0))]

    # Page 3 item B: the body text that should be attached to the bookmark.
    item_page3_body = MagicMock()
    item_page3_body.text = "If the paper jams, open the rear cover."
    item_page3_body.prov = [MagicMock(page_no=3, bbox=MagicMock(t=700.0))]

    doc.texts = [item_page2, item_page3_title, item_page3_body]

    chunks = chunk_text_by_coordinates(doc, mock_manual)

    assert bm.page_top == 750.0

    matching = [c for c in chunks if "If the paper jams" in c["text"]]
    assert len(matching) == 1
    assert matching[0]["metadata"]["bookmark_id"] == "bm_troubleshooting"
