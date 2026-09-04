"""Chunking of a converted Docling document into bookmark-aware chunks.

The document is walked in reading order via ``doc.iterate_items()``. Text items
are buffered per manual bookmark, while tables and pictures become dedicated
chunks. A figure chunk also carries the caption, the labels drawn inside the
picture and the Docling description in its metadata, so the builder can persist
them next to the image. Docling types are never imported here: item kinds are
detected through their ``label`` value and duck-typed methods, so tests can use
light fakes.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from mcp_manual_walker.config import settings
from mcp_manual_walker.models import Bookmark, Manual

logger = logging.getLogger(__name__)

# Docling item labels that need special handling; anything else is text.
TABLE_LABEL = "table"
PICTURE_LABEL = "picture"
CAPTION_LABEL = "caption"
HEADING_LABELS = frozenset({"section_header", "title"})

# A bookmark destination often sits a few points above its heading, so the
# coordinate rule allows a small tolerance (PDF points, bottom-left origin).
BOOKMARK_TOP_TOLERANCE = 10.0

# Leading section numbering to drop when comparing titles: "3", "2.1", "3-1",
# "IV.", "A." (letters and roman numerals require trailing punctuation, so an
# ordinary leading word is never mistaken for a numbering prefix).
_NUMBERING_PREFIX_RE = re.compile(
    r"^(?:\d+(?:[.\-]\d+)*[.)]?|[ivxlcdm]+[.)]|[a-z][.)])\s+"
)
_TITLE_PUNCTUATION = " \t.,:;!?-–—_()[]{}<>\"'`*#|/\\、。，．：；・"

# Minimum length of the shorter normalized title for containment matching.
_MIN_CONTAINMENT_LEN = 3


def _split_if_long(content: str, splitter: Any) -> List[str]:
    """Splits ``content`` only when it exceeds CHUNK_SIZE.

    Returning the text untouched below the limit matters: the splitter is
    allowed to re-join and re-wrap, so running everything through it would
    change chunks that were already the right size.
    """
    if len(content) <= settings.CHUNK_SIZE:
        return [content]
    parts = splitter.split_text(content)
    return parts or [content]


def _label_value(item: Any) -> str:
    """Returns the item label as a plain string (DocItemLabel or str)."""
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _item_text(item: Any) -> Optional[str]:
    """Returns the item text when the item carries one, else None."""
    text = getattr(item, "text", None)
    return text if isinstance(text, str) else None


def _first_prov(item: Any) -> Optional[Any]:
    """Returns the first provenance entry of an item, or None."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    return prov[0]


def _resolve(ref: Any, doc: Any) -> Optional[Any]:
    """Resolves a RefItem against the document; passes plain items through."""
    if ref is None:
        return None
    resolve = getattr(ref, "resolve", None)
    if callable(resolve):
        try:
            return resolve(doc)
        except Exception:  # noqa: BLE001 - a broken ref must not stop chunking
            return None
    if hasattr(ref, "label"):
        return ref
    return None


def _normalize_title(s: str) -> str:
    """Normalizes a heading/bookmark title for comparison."""
    if not s:
        return ""
    normalized = s.strip().lower()
    normalized = _NUMBERING_PREFIX_RE.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(_TITLE_PUNCTUATION)


def _page_top_key(bm: Bookmark) -> float:
    """Sort key placing the bookmarks of a page top-down (missing tops last)."""
    return bm.page_top if bm.page_top is not None else -1.0


def _ordering_key(bm: Bookmark) -> int:
    """Sort key over Bookmark.ordering, tolerant of missing values."""
    ordering = getattr(bm, "ordering", None)
    return ordering if isinstance(ordering, int) else 0


def _match_bookmark_by_title(
    page_bookmarks: List[Bookmark], text: str
) -> Optional[Bookmark]:
    """Finds the bookmark on a page whose title matches a heading text."""
    target = _normalize_title(text or "")
    if not target:
        return None

    ordered = sorted(page_bookmarks, key=_ordering_key)

    for bm in ordered:
        if _normalize_title(bm.title) == target:
            return bm

    for bm in ordered:
        title = _normalize_title(bm.title)
        if not title:
            continue
        if min(len(title), len(target)) < _MIN_CONTAINMENT_LEN:
            continue
        if title in target or target in title:
            return bm

    return None


def _caption_belongs_to_figure(item: Any, doc: Any) -> bool:
    """True when a caption item is attached to a picture or a table."""
    parent = getattr(item, "parent", None)
    if parent is None:
        return False

    owner = _resolve(parent, doc)
    if owner is not None:
        return _label_value(owner) in (PICTURE_LABEL, TABLE_LABEL)

    cref = str(getattr(parent, "cref", ""))
    return cref.startswith("#/pictures/") or cref.startswith("#/tables/")


def _caption_of(item: Any, doc: Any) -> str:
    """Returns the joined caption text of a table/picture item ("" if none)."""
    caption_text = getattr(item, "caption_text", None)
    if not callable(caption_text):
        return ""
    try:
        return (caption_text(doc) or "").strip()
    except Exception:  # noqa: BLE001 - a caption is never worth failing on
        return ""


def _make_table_exporter(doc: Any):
    """Returns a function exporting one table to markdown.

    `TableItem.export_to_markdown(doc)` builds a `MarkdownDocSerializer` per
    call, and constructing one revalidates the whole document: pydantic runs
    `validate_document`, which clamps every table cell's bounding box on every
    page. Cost is therefore (tables x cells in the document), and on a
    table-dense manual that is the dominant cost of the entire build -- a
    490-page font reference spent over eleven minutes there, with the parent's
    RSS climbing from 6 GB to 14.6 GB, after a 46-minute conversion.

    Building the serializer once per document makes it linear. Measured on a
    100-page document with cells carrying provenance, output identical:

        tables   cells    per-table    shared
            25   4,000       0.58 s    0.07 s
           100  16,000       8.15 s    0.31 s
           200  32,000      31.82 s    0.61 s

    Falls back to the per-call path if the serializer cannot be built, so a
    docling_core that moves it costs speed rather than the build.
    """
    try:
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        serializer = MarkdownDocSerializer(doc=doc)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Falling back to per-table markdown export (%s); this is "
            "quadratic in the number of tables.",
            e,
        )
        return lambda item: _table_markdown(item, doc)

    def export(item: Any) -> str:
        try:
            return (serializer.serialize(item=item).text or "").strip()
        except Exception:  # noqa: BLE001 - one bad table is not a document
            return _table_markdown(item, doc)

    return export


def _table_markdown(item: Any, doc: Any) -> str:
    """Exports a table item to markdown, one serializer per call."""
    export = getattr(item, "export_to_markdown", None)
    if not callable(export):
        return ""
    try:
        return (export(doc) or "").strip()
    except TypeError:
        return (export() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _picture_labels(item: Any, doc: Any, caption: str) -> List[str]:
    """Collects the text labels drawn inside a picture, in reading order."""
    labels: List[str] = []
    for ref in getattr(item, "children", None) or []:
        child = _resolve(ref, doc)
        if child is None or _label_value(child) == CAPTION_LABEL:
            continue
        text = (_item_text(child) or "").strip()
        if not text or text == caption:
            continue
        labels.append(text)
    return labels


def _picture_description(item: Any) -> str:
    """Returns the Docling-generated picture description, if any."""
    meta = getattr(item, "meta", None)
    description = getattr(meta, "description", None) if meta is not None else None
    text = getattr(description, "text", None) if description is not None else None
    return text.strip() if isinstance(text, str) else ""


def _picture_index(item: Any, fallback: int) -> int:
    """Parses the picture index out of a self_ref like "#/pictures/3"."""
    self_ref = str(getattr(item, "self_ref", "") or "")
    tail = self_ref.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else fallback


def chunk_document(doc, manual: Manual) -> List[Dict[str, Any]]:
    """
    Chunks a Docling document in reading order.

    Items are attached to the manual's bookmarks by matching detected section
    headers against the bookmark titles first, falling back to the page and
    Y-coordinate of the item. Tables and pictures get their own chunks.

    Returns a list of ``{"text": str, "metadata": {...}}`` dicts whose metadata
    always carries ``manual_id``, ``bookmark_id`` and ``type``, plus ``page``
    for table/figure chunks. Figure chunks additionally carry
    ``picture_index``, ``figure_caption``, ``figure_labels`` (comma-joined,
    ``""`` when the picture has none) and ``figure_description``.
    """
    chunks: List[Dict[str, Any]] = []

    if not hasattr(doc, "iterate_items"):
        text = doc.export_to_markdown()
        return [
            {
                "text": text,
                "metadata": {
                    "manual_id": manual.id,
                    "bookmark_id": None,
                    "type": "text",
                },
            }
        ]

    # One serializer for the whole document: see _make_table_exporter.
    export_table = _make_table_exporter(doc)

    # Markdown-aware splitter, so table rows and headings survive a split.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    def make_metadata(
        bookmark: Optional[Bookmark], chunk_type: str, **extra: Any
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "manual_id": manual.id,
            "bookmark_id": bookmark.id if bookmark else None,
            "type": chunk_type,
        }
        metadata.update(extra)
        return metadata

    def add_chunk(
        content: str, bookmark: Optional[Bookmark], chunk_type: str, **extra: Any
    ) -> None:
        if not content:
            return
        for sub_chunk in splitter.split_text(content):
            chunks.append(
                {
                    "text": sub_chunk,
                    "metadata": make_metadata(bookmark, chunk_type, **extra),
                }
            )

    raw_bookmarks = list(manual.bookmarks)

    # Group text items by page once, so the fallback lookup below and the
    # coordinate rule stay cheap on large documents.
    texts_by_page: Dict[int, List[Any]] = {}
    for item, _level in doc.iterate_items(with_groups=False):
        if _label_value(item) in (TABLE_LABEL, PICTURE_LABEL):
            continue
        if _item_text(item) is None:
            continue
        prov = _first_prov(item)
        if prov is None:
            continue
        texts_by_page.setdefault(prov.page_no, []).append(item)

    # Bookmarks whose destination carries no Y coordinate are resolved by
    # looking for their title on their own (1-based) page.
    for bm in raw_bookmarks:
        if bm.page_top is not None:
            continue
        for item in texts_by_page.get(bm.page_num, []):
            if bm.title.lower() in (_item_text(item) or "").lower():
                prov = _first_prov(item)
                bm.page_top = getattr(prov.bbox, "t", 0.0)
                logger.info(
                    f"Resolved missing coordinate for bookmark '{bm.title}' "
                    f"on page {bm.page_num} to {bm.page_top}"
                )
                break

    bms_by_page: Dict[int, List[Bookmark]] = {}
    for bm in raw_bookmarks:
        bms_by_page.setdefault(bm.page_num, []).append(bm)

    # Sort each page's bookmarks by Top DESC for the scanning logic
    for page_bms in bms_by_page.values():
        page_bms.sort(key=_page_top_key, reverse=True)

    current_bookmark: Optional[Bookmark] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        content = "\n\n".join(buffer)
        buffer = []
        add_chunk(content, current_bookmark, "text")

    picture_count = 0

    for item, _level in doc.iterate_items(with_groups=False):
        kind = _label_value(item)
        text = _item_text(item)

        prov = _first_prov(item)
        if prov is None:
            # No coordinates: keep the text under the bookmark in effect.
            if text and text.strip():
                buffer.append(text.strip())
            continue

        page_no = prov.page_no
        item_top = getattr(prov.bbox, "t", 0.0)
        page_bms = bms_by_page.get(page_no, [])

        candidate: Optional[Bookmark] = None
        if kind in HEADING_LABELS and text:
            candidate = _match_bookmark_by_title(page_bms, text)
            if candidate is not None and candidate.page_top is None:
                candidate.page_top = item_top
                # The page list must stay sorted top-down, otherwise the early
                # break of the coordinate rule below cuts the scan short for
                # the items that follow on this page.
                page_bms.sort(key=_page_top_key, reverse=True)

        if candidate is None:
            # Coordinate rule: the last bookmark that starts above this item.
            for bm in page_bms:
                bm_top = bm.page_top if bm.page_top is not None else -1.0
                if item_top < (bm_top + BOOKMARK_TOP_TOLERANCE):
                    candidate = bm
                else:
                    break

        if candidate is not None and candidate is not current_bookmark:
            flush()
            current_bookmark = candidate

        if kind == TABLE_LABEL:
            flush()
            caption = _caption_of(item, doc)
            markdown = export_table(item)
            content = "\n\n".join(part for part in (caption, markdown) if part)
            add_chunk(content, current_bookmark, "table", page=page_no)
            continue

        if kind == PICTURE_LABEL:
            flush()
            caption = _caption_of(item, doc)
            labels = _picture_labels(item, doc, caption)
            description = _picture_description(item)
            parts = [
                caption,
                "Labels: " + ", ".join(labels) if labels else "",
                description,
            ]
            content = "\n\n".join(part for part in parts if part)
            if not content:
                content = f"Figure on page {page_no}"
            index = _picture_index(item, picture_count)
            picture_count += 1
            # A figure chunk is usually small and self-contained, so it is kept
            # whole -- but "usually" is not "always": the OCR'd labels of a
            # dense form ran to 6943 characters on one real manual, three and a
            # half times CHUNK_SIZE. Left whole that becomes a 3836-token chunk
            # against an EMBEDDING_MAX_SEQ_LENGTH of 4096, so a slightly bigger
            # figure would be truncated with no warning at all, and it drags
            # every batch it is embedded in up to its own length. Split only
            # the ones that are actually too long; the common case is
            # unaffected, and every piece keeps the same picture_index so they
            # all resolve to one figure row.
            for part in _split_if_long(content, splitter):
                chunks.append(
                    {
                        "text": part,
                        "metadata": make_metadata(
                            current_bookmark,
                            "figure",
                            page=page_no,
                            picture_index=index,
                            figure_caption=caption,
                            figure_labels=", ".join(labels),
                            figure_description=description,
                        ),
                    }
                )
            continue

        if kind == CAPTION_LABEL and _caption_belongs_to_figure(item, doc):
            # Already emitted as part of its table/figure chunk.
            continue

        if text and text.strip():
            buffer.append(text.strip())

    flush()

    return chunks
