"""Tests for the reading-order chunker.

The fakes below imitate just enough of the Docling document model: items expose
a ``label`` with a ``.value``, optional ``text`` / ``prov`` / ``children`` and
the duck-typed ``caption_text`` / ``export_to_markdown`` methods.
"""

from types import SimpleNamespace

import pytest

from mcp_manual_walker.chunking import (
    _match_bookmark_by_title,
    _normalize_title,
    chunk_document,
)
from mcp_manual_walker.models import Bookmark


class Label:
    """Stand-in for docling's DocItemLabel str-enum."""

    def __init__(self, value: str):
        self.value = value


class BBox:
    def __init__(self, top: float):
        self.t = top
        self.b = top - 10.0
        self.l = 0.0  # noqa: E741 - mirrors the docling attribute name
        self.r = 100.0


class Prov:
    def __init__(self, page_no: int, top: float):
        self.page_no = page_no
        self.bbox = BBox(top)


class Ref:
    """Stand-in for docling's RefItem."""

    def __init__(self, target, cref: str = ""):
        self._target = target
        self.cref = cref

    def resolve(self, doc):
        return self._target


class TextItem:
    def __init__(self, text, page=None, top=0.0, label="text", parent=None):
        self.label = Label(label)
        self.text = text
        self.prov = [Prov(page, top)] if page is not None else []
        self.parent = parent


class TableItem:
    def __init__(self, page, top, markdown, caption=""):
        self.label = Label("table")
        self.prov = [Prov(page, top)]
        self._markdown = markdown
        self._caption = caption

    def caption_text(self, doc):
        return self._caption

    def export_to_markdown(self, doc=None):
        return self._markdown


class PictureItem:
    def __init__(self, page, top, index, caption="", description=None):
        self.label = Label("picture")
        self.prov = [Prov(page, top)]
        self.self_ref = f"#/pictures/{index}"
        self.children = []
        self._caption = caption
        self.meta = None
        if description is not None:
            self.meta = SimpleNamespace(
                description=SimpleNamespace(text=description)
            )

    def caption_text(self, doc):
        return self._caption


class FakeDoc:
    """Document exposing items in reading order, as iterate_items() does."""

    def __init__(self, items, pictures=None):
        self.items = items
        self.pictures = pictures or [i for i in items if _is_picture(i)]

    def iterate_items(self, **kwargs):
        return iter([(item, 0) for item in self.items])

    def export_to_markdown(self):
        return "Full Generic Text"


def _is_picture(item):
    return getattr(getattr(item, "label", None), "value", None) == "picture"


def make_bookmark(bm_id, title, page_num, page_top, ordering=0):
    return Bookmark(
        id=bm_id,
        manual_id="test_manual_id",
        ordering=ordering,
        title=title,
        level=1,
        page_num=page_num,
        page_top=page_top,
    )


@pytest.fixture
def manual():
    return SimpleNamespace(id="test_manual_id", bookmarks=[])


def test_normalize_title_drops_numbering_and_punctuation():
    assert _normalize_title("3 Installation") == "installation"
    assert _normalize_title("2.1  Setup steps") == "setup steps"
    assert _normalize_title("3-1 Wiring") == "wiring"
    assert _normalize_title("IV. Overview") == "overview"
    assert _normalize_title("A. Appendix:") == "appendix"
    assert _normalize_title("A quick guide") == "a quick guide"


def test_match_bookmark_by_title_prefers_exact_match():
    exact = make_bookmark("bm_exact", "Setup", 1, None, ordering=1)
    contained = make_bookmark("bm_contained", "Setup steps", 1, None, ordering=0)

    assert _match_bookmark_by_title([contained, exact], "1 Setup") is exact
    assert _match_bookmark_by_title([contained], "1 Setup") is contained
    # Too short to be matched by containment
    assert _match_bookmark_by_title([make_bookmark("b", "AB", 1, None)], "ABC") is None


def test_chunking_fallback_no_prov(manual):
    """Items without provenance are still buffered into a chunk."""
    doc = FakeDoc([TextItem("Start Text"), TextItem("More Text")])

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 1
    assert "Start Text" in chunks[0]["text"]
    assert "More Text" in chunks[0]["text"]
    assert chunks[0]["metadata"]["manual_id"] == "test_manual_id"
    assert chunks[0]["metadata"]["bookmark_id"] is None
    assert chunks[0]["metadata"]["type"] == "text"


def test_chunking_with_bookmarks(manual):
    """Coordinate mapping plus context switch, including a page without any."""
    bm1 = make_bookmark("bm1", "Header 1", 1, 800.0, ordering=0)
    bm2 = make_bookmark("bm2", "Header 2", 1, 600.0, ordering=1)
    manual.bookmarks = [bm1, bm2]

    doc = FakeDoc(
        [
            # Below Header 1 (800), above Header 2 (600) -> bm1
            TextItem("Content for Header 1", page=1, top=750.0),
            # Below Header 2 (600) -> bm2
            TextItem("Content for Header 2", page=1, top=550.0),
            # Page 2 has no bookmarks -> the current one persists
            TextItem("Content on Page 2", page=2, top=800.0),
        ]
    )

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 2

    assert chunks[0]["metadata"]["bookmark_id"] == "bm1"
    assert "Content for Header 1" in chunks[0]["text"]

    assert chunks[1]["metadata"]["bookmark_id"] == "bm2"
    assert "Content for Header 2" in chunks[1]["text"]
    assert "Content on Page 2" in chunks[1]["text"]


def test_chunking_no_iterate_items(manual):
    """Documents that cannot be walked fall back to the full markdown."""
    doc = SimpleNamespace(export_to_markdown=lambda: "Full Generic Text")

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Full Generic Text"
    assert chunks[0]["metadata"]["bookmark_id"] is None
    assert chunks[0]["metadata"]["type"] == "text"


def test_chunking_large_content(manual):
    """Large content is split by the LangChain markdown splitter."""
    large_text = "A" * 1500 + "\n\n" + "B" * 1000
    doc = FakeDoc([TextItem(large_text)])

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 2

    full_text_out = "".join(c["text"] for c in chunks)
    assert "AAAA" in full_text_out
    assert "BBBB" in full_text_out

    assert chunks[0]["metadata"]["manual_id"] == "test_manual_id"
    assert chunks[1]["metadata"]["manual_id"] == "test_manual_id"


def test_chunking_sequence_and_overlap(manual):
    """Continuous text is split with the configured overlap."""
    doc = FakeDoc([TextItem("P" * 2500)])

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 2
    assert len(chunks[0]["text"]) == 2000
    assert len(chunks[1]["text"]) == 700

    assert chunks[0]["text"][-200:] == chunks[1]["text"][:200]
    assert chunks[0]["metadata"] == chunks[1]["metadata"]


def test_chunking_fallback_resolves_missing_bookmark_coordinate(manual):
    """A bookmark without page_top is resolved on its own (1-based) page."""
    bm = make_bookmark("bm_troubleshooting", "Troubleshooting", 3, None)
    manual.bookmarks = [bm]

    doc = FakeDoc(
        [
            # Same top as the page-3 title but on the wrong page: guards
            # against the old off-by-one (page_num - 1) lookup.
            TextItem("Unrelated heading", page=2, top=750.0),
            TextItem("3 Troubleshooting", page=3, top=750.0, label="section_header"),
            TextItem(
                "If the paper jams, open the rear cover.", page=3, top=700.0
            ),
        ]
    )

    chunks = chunk_document(doc, manual)

    assert bm.page_top == 750.0

    matching = [c for c in chunks if "If the paper jams" in c["text"]]
    assert len(matching) == 1
    assert matching[0]["metadata"]["bookmark_id"] == "bm_troubleshooting"


def test_section_header_attaches_bookmark_without_coordinate(manual):
    """A "3 Installation" header attaches the section to bookmark "Installation"."""
    bm = make_bookmark("bm_install", "Installation", 1, None)
    manual.bookmarks = [bm]

    doc = FakeDoc(
        [
            TextItem("Printer Manual", page=1, top=900.0, label="title"),
            TextItem("3 Installation", page=1, top=700.0, label="section_header"),
            TextItem("Place the unit on a flat surface.", page=1, top=650.0),
        ]
    )

    chunks = chunk_document(doc, manual)

    body = [c for c in chunks if "flat surface" in c["text"]]
    assert len(body) == 1
    assert body[0]["metadata"]["bookmark_id"] == "bm_install"
    assert body[0]["metadata"]["type"] == "text"
    assert bm.page_top == 700.0


def test_section_header_matches_numbered_bookmark_title(manual):
    """Title matching ignores numbering, where plain containment fails."""
    bm = make_bookmark("bm_install", "3. Installation", 1, None)
    manual.bookmarks = [bm]

    doc = FakeDoc(
        [
            TextItem("3 Installation", page=1, top=700.0, label="section_header"),
            TextItem("Place the unit on a flat surface.", page=1, top=650.0),
        ]
    )

    chunks = chunk_document(doc, manual)

    # The pre-pass cannot resolve "3. Installation" inside "3 Installation",
    # so only the header title match can attach these items.
    assert bm.page_top == 700.0
    assert all(c["metadata"]["bookmark_id"] == "bm_install" for c in chunks)


def test_table_becomes_its_own_chunk(manual):
    """Tables get a dedicated chunk carrying caption, page and section."""
    bm = make_bookmark("bm_specs", "Specifications", 2, 800.0)
    manual.bookmarks = [bm]

    table = TableItem(
        page=2,
        top=600.0,
        markdown="| Item | Value |\n|---|---|\n| Weight | 5 kg |",
        caption="Table 2-1: Ratings",
    )
    doc = FakeDoc(
        [
            TextItem("Specifications", page=2, top=800.0, label="section_header"),
            TextItem("The ratings are listed below.", page=2, top=750.0),
            table,
        ]
    )

    chunks = chunk_document(doc, manual)

    tables = [c for c in chunks if c["metadata"]["type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["text"].startswith("Table 2-1: Ratings")
    assert "| Weight | 5 kg |" in tables[0]["text"]
    assert tables[0]["metadata"]["page"] == 2
    assert tables[0]["metadata"]["bookmark_id"] == "bm_specs"

    # The surrounding text is flushed into its own chunk, before the table.
    text_chunks = [c for c in chunks if c["metadata"]["type"] == "text"]
    assert len(text_chunks) == 1
    assert "ratings are listed below" in text_chunks[0]["text"]
    assert chunks.index(text_chunks[0]) < chunks.index(tables[0])


def test_picture_becomes_figure_chunk(manual):
    """Figures carry caption, labels and description, and are not duplicated."""
    bm = make_bookmark("bm_panel", "Front panel", 2, 800.0)
    manual.bookmarks = [bm]

    picture = PictureItem(
        page=2,
        top=600.0,
        index=3,
        caption="Figure 3-1: Controller wiring",
        description="A wiring diagram of the controller.",
    )
    caption_item = TextItem(
        "Figure 3-1: Controller wiring",
        page=2,
        top=520.0,
        label="caption",
        parent=Ref(picture, cref="#/pictures/3"),
    )
    label_a = TextItem("Controller", page=2, top=590.0)
    label_b = TextItem("Sensor", page=2, top=570.0)
    picture.children = [Ref(caption_item), Ref(label_a), Ref(label_b)]

    doc = FakeDoc(
        [
            TextItem("Front panel", page=2, top=800.0, label="section_header"),
            TextItem("The front panel is shown below.", page=2, top=750.0),
            picture,
            caption_item,
        ]
    )

    chunks = chunk_document(doc, manual)

    figures = [c for c in chunks if c["metadata"]["type"] == "figure"]
    assert len(figures) == 1
    figure = figures[0]
    assert figure["text"] == (
        "Figure 3-1: Controller wiring\n\n"
        "Labels: Controller, Sensor\n\n"
        "A wiring diagram of the controller."
    )
    assert figure["metadata"]["page"] == 2
    assert figure["metadata"]["picture_index"] == 3
    assert figure["metadata"]["bookmark_id"] == "bm_panel"

    # The caption item must not be repeated in a text chunk.
    text_chunks = [c for c in chunks if c["metadata"]["type"] == "text"]
    assert all("Figure 3-1" not in c["text"] for c in text_chunks)


def test_picture_without_caption_labels_or_description(manual):
    """A bare picture still produces a placeholder figure chunk."""
    manual.bookmarks = []
    picture = PictureItem(page=4, top=500.0, index=0)
    doc = FakeDoc([picture])

    chunks = chunk_document(doc, manual)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Figure on page 4"
    assert chunks[0]["metadata"]["type"] == "figure"
    assert chunks[0]["metadata"]["page"] == 4
    assert chunks[0]["metadata"]["picture_index"] == 0
    assert chunks[0]["metadata"]["bookmark_id"] is None
