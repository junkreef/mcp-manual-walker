import argparse
import contextlib
import ctypes
import fnmatch
import io
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import uuid
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

# Imports for dependencies
from sqlalchemy import select
from sqlalchemy.orm import Session

from mcp_manual_walker import progress

# Suppress warnings from libraries
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("builder")


try:
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        HeadingHierarchyOptions,
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling_core.types.doc import ImageRefMode
    # We might need specific options if we want to speed up or customize
except ImportError as e:
    logger.error(f"Failed to import docling: {e}", exc_info=True)
    DocumentConverter = None
    StandardPdfPipeline = None

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except ImportError:
    MarkdownHeaderTextSplitter = None

try:
    import chromadb
except ImportError:
    chromadb = None

# Imports for DB Sync
# Local imports
try:
    from mcp_manual_walker.chunking import _picture_description, chunk_document
    from mcp_manual_walker.config import settings
    from mcp_manual_walker.database import SessionLocal, init_db
    from mcp_manual_walker.embeddings import (
        COLLECTION_NAME,
        check_collection_model,
        collection_metadata,
        get_embedder,
    )
    from mcp_manual_walker.models import Bookmark, Figure, Manual
    from mcp_manual_walker.pdf_utils import extract_pdf_fingerprint
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    sys.exit(1)


# Chroma rejects very large single batches, so inserts are sliced.
CHROMA_ADD_BATCH_SIZE = 1000

# Per-process Docling converter, initialized once in each worker process.
_converter = None


def check_dependencies():
    missing = []
    if DocumentConverter is None:
        missing.append("docling")
    if MarkdownHeaderTextSplitter is None:
        missing.append("langchain-text-splitters")
    if chromadb is None:
        missing.append("chromadb")

    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install them with: uv sync --extra builder")
        sys.exit(1)


def sync_manual_to_db(
    session: Session,
    pdf_path: Path,
    pdf_root: Path,
    file_hash: str,
    metadata: dict,
) -> tuple[Manual, bool, bool]:
    """
    Syncs the Manual and Bookmarks to the SQLite DB.

    The file hash and the pypdf metadata are computed beforehand (possibly in a
    worker process) and passed in, so this function only touches the DB.

    Returns:
        (Manual, updated, existed): the manual object, whether it was created or
        refreshed (True) or left unchanged (False), and whether a row for this
        relative path already existed before the call.
    """
    # Use consistent relative path from the root PDF directory
    try:
        rel_path_str = str(pdf_path.relative_to(pdf_root))
    except ValueError:
        # Fallback if path is not relative to root (should be rare in current usage)
        rel_path_str = pdf_path.name

    stmt = select(Manual).where(Manual.relative_path == rel_path_str)
    manual = session.execute(stmt).scalars().first()
    existed = manual is not None

    if manual:
        logger.info(f"Manual {rel_path_str} found in DB. Checking hash...")
        if manual.file_hash == file_hash:
            logger.info("Hash match. Skipping DB sync (bookmarks).")
            return manual, False, existed
        else:
            logger.info("Hash mismatch. Updating...")
            # Delete old bookmarks and figures: they describe the old content.
            manual.bookmarks.clear()
            manual.figures.clear()
    else:
        logger.info(f"Creating new Manual entry for {rel_path_str}")
        manual = Manual(id=str(uuid.uuid4()))
        session.add(manual)

    # Update attributes
    manual.file_name = pdf_path.name
    manual.document_title = metadata.get("document_title")
    manual.relative_path = rel_path_str
    manual.file_hash = file_hash
    manual.page_count = metadata.get("page_count", 0)

    # Insert Bookmarks
    bookmarks_data = metadata.get("bookmarks", [])

    # Stack to track parents: [(level, bookmark_object)]
    parent_stack = []

    for idx, bm_data in enumerate(bookmarks_data):
        bm_id = str(uuid.uuid4())
        level = bm_data["level"]

        # Determine parent
        parent_id = None

        # Pop stack until we find a parent with level < current level
        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()

        if parent_stack:
            parent_id = parent_stack[-1][1].id

        new_bm = Bookmark(
            id=bm_id,
            manual_id=manual.id,
            ordering=idx,
            title=bm_data["title"],
            level=level,
            page_num=bm_data["page_num"],
            page_top=bm_data.get("top"),  # From our enhanced pdf_utils
            parent_id=parent_id,
        )
        session.add(new_bm)

        # Push to stack
        parent_stack.append((level, new_bm))

    session.commit()
    logger.info(f"Synced {len(bookmarks_data)} bookmarks to DB.")
    return manual, True, existed


def _make_process_executor(
    max_workers: int, initializer=None, initargs=()
) -> ProcessPoolExecutor:
    """
    Creates a process pool using the "spawn" start method.

    This is the single place where a ProcessPoolExecutor is built, so tests can
    swap it for a thread pool and keep everything inside one process.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )


def _picture_description_options():
    """
    Builds the Docling options for describing figures via a vision model.

    Returns None when PICTURE_DESCRIPTION_URL is unset, which is how the
    feature stays off by default.
    """
    if not settings.PICTURE_DESCRIPTION_URL:
        return None

    params: dict = {"max_tokens": settings.PICTURE_DESCRIPTION_MAX_TOKENS}
    if settings.PICTURE_DESCRIPTION_MODEL:
        params["model"] = settings.PICTURE_DESCRIPTION_MODEL

    headers: dict = {}
    if settings.PICTURE_DESCRIPTION_API_KEY:
        headers["Authorization"] = f"Bearer {settings.PICTURE_DESCRIPTION_API_KEY}"

    return PictureDescriptionApiOptions(
        url=settings.PICTURE_DESCRIPTION_URL,
        params=params,
        headers=headers,
        prompt=settings.PICTURE_DESCRIPTION_PROMPT,
        timeout=settings.PICTURE_DESCRIPTION_TIMEOUT,
        concurrency=settings.PICTURE_DESCRIPTION_CONCURRENCY,
        scale=settings.DOCLING_IMAGES_SCALE,
        picture_area_threshold=settings.PICTURE_DESCRIPTION_AREA_THRESHOLD,
    )


def _create_converter(num_threads: int):
    """Builds a DocumentConverter configured for the accelerator settings."""
    pipeline_options = PdfPipelineOptions()
    # AcceleratorOptions accepts the device as a plain string ("auto", "cuda:0", ...)
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=settings.DOCLING_DEVICE,
        num_threads=num_threads,
    )
    # Increase batch sizes to improve GPU utilization
    pipeline_options.ocr_batch_size = settings.DOCLING_OCR_BATCH_SIZE
    pipeline_options.layout_batch_size = settings.DOCLING_LAYOUT_BATCH_SIZE
    pipeline_options.table_batch_size = settings.DOCLING_TABLE_BATCH_SIZE
    # Bound the pages in flight. Every page in a queue holds its rendered image
    # until the assemble stage releases it, so this caps the peak far more
    # effectively than the page count suggests.
    pipeline_options.queue_max_size = settings.DOCLING_QUEUE_MAX_SIZE

    # Derive section-header levels from PDF bookmarks / numbering / font style.
    #
    # The pipeline stage itself is switched off whenever this build can do the
    # assignment itself, because a page range must not be levelled on its own:
    # the numbering and style signals rank a heading against the rest of the
    # document. Instead every conversion -- split or not -- is levelled once
    # over the finished document, by `assign_heading_levels`. The options
    # object still carries the settings that pass uses.
    #
    # It cannot be toggled per call: the converter caches pipelines by an
    # options hash, so flipping `enabled` would build a second pipeline and
    # load a second copy of the models into VRAM.
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=_SPLIT_SUPPORT is None
    )
    # The font-style signal reads the parsed PDF cells, which the pipeline
    # discards unless they are explicitly kept.
    #
    # Keeping them is the single largest contributor to a worker's resident
    # memory (~1.09 MB per page of the document being converted), and nothing
    # in this repository reads `SectionHeaderItem.level`: chunking matches on
    # the heading *label* only. Do not conclude from that grep that the levels
    # are dead weight. They set the "#" depth of the headings inside every
    # exported chunk body, and the RAG that reads those chunks depends on that
    # structure to summarise them. Turning either option off is an output
    # change, not an optimisation.
    pipeline_options.generate_parsed_pages = True

    # Render the detected pictures: the crops are persisted as PNG blobs in the
    # SQLite figures table (and written next to the markdown dump on request).
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = settings.DOCLING_IMAGES_SCALE

    pipeline_options.ocr_options = RapidOcrOptions(
        backend=settings.DOCLING_OCR_BACKEND,
        lang=[settings.DOCLING_OCR_LANG],
    )

    picture_description_options = _picture_description_options()
    if picture_description_options is not None:
        # enable_remote_services unlocks Docling pipelines that call an
        # external HTTP endpoint; here that endpoint is the user's own local
        # vision model server, not a third-party cloud service.
        pipeline_options.enable_remote_services = True
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = picture_description_options

    # The default ThreadedDoclingParseDocumentBackend (docling-parse's native
    # threaded parser) was observed dropping pages ("Page N failed to parse",
    # PARTIAL_SUCCESS) and segfaulting the worker; cross-document parallelism
    # already comes from DOCLING_WORKERS, so the single-threaded parser is
    # used here for deterministic output.
    pdf_format_option = PdfFormatOption(
        pipeline_options=pipeline_options,
        backend=DoclingParseDocumentBackend,
    )
    pipeline_cls = _make_reporting_pipeline_cls()
    if pipeline_cls is not None:
        pdf_format_option.pipeline_cls = pipeline_cls

    return DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})


# Pages between progress events while a document converts. Small enough that a
# long manual visibly moves, large enough that the event log stays a rounding
# error next to the conversion itself (~17k events for a 172k-page corpus).
_PAGE_REPORT_STEP = 10

# Shared with the worker processes through the pool initializer, and held by
# the parent while it embeds. None means "no limit".
_gpu_slot = None


@contextlib.contextmanager
def gpu_slot(what: str, role: str, path: str, **fields):
    """Holds one of the GPU slots for the duration of the block.

    The Docling workers and the embedding model share one device and neither
    yields to the other, so without this they simply race to exhaust it. The
    slot is coarse on purpose: it is held for a whole conversion or a whole
    embedding call, which is long, but the alternative is a lock around every
    allocation, and there is no way to reach inside Docling or torch for that.

    Holding one is also reported, so the monitor can show what the semaphore is
    doing rather than leaving it to be inferred from which documents move.
    """
    waited = time.monotonic()
    if _gpu_slot is not None:
        _gpu_slot.acquire()
        delay = time.monotonic() - waited
        if delay > 1.0:
            logger.info(f"Waited {delay:.0f}s for a GPU slot before {what}.")
    progress.emit(
        "slot", owner=os.getpid(), role=role, path=path, state="start", **fields
    )
    try:
        yield
    finally:
        progress.emit("slot", owner=os.getpid(), state="end")
        if _gpu_slot is not None:
            _gpu_slot.release()

# The document this worker process is converting. A worker converts one at a
# time, so a module global is the whole state that is needed -- but the page
# callback runs on the pipeline's assemble thread, hence the lock.
_page_lock = threading.Lock()
_current_document: str | None = None
_current_part = 0
_pages_converted = 0


def _begin_document(rel_path: str, part: int = 1) -> None:
    global _current_document, _current_part, _pages_converted
    with _page_lock:
        _current_document = rel_path
        _current_part = part
        _pages_converted = 0


def _end_document() -> None:
    global _current_document
    with _page_lock:
        _current_document = None


def _report_page() -> None:
    """Called once per page, as that page leaves the last pipeline stage."""
    global _pages_converted
    with _page_lock:
        _pages_converted += 1
        count = _pages_converted
        document = _current_document
        part = _current_part
    if document and (count == 1 or count % _PAGE_REPORT_STEP == 0):
        # The part is part of the key: several parts of one document report
        # concurrently, and the reader sums them rather than taking a maximum.
        progress.emit("page", path=document, part=part, pages_done=count)


def _make_reporting_pipeline_cls():
    """A pipeline that reports pages, or None if the hook is not there.

    `_release_page_resources` is Docling's own per-page postprocess on the
    final stage -- the one place that sees every page exactly once, after it
    is finished. It is internal API, so if a Docling upgrade renames it this
    returns None and the build runs on the stock pipeline without per-page
    progress, rather than failing.
    """
    if StandardPdfPipeline is None:
        return None
    if not hasattr(StandardPdfPipeline, "_release_page_resources"):
        logger.warning(
            "Docling's StandardPdfPipeline._release_page_resources is gone; "
            "per-page progress is disabled."
        )
        return None

    class ProgressReportingPdfPipeline(StandardPdfPipeline):
        def _release_page_resources(self, item):
            super()._release_page_resources(item)
            _report_page()

    return ProgressReportingPdfPipeline


def _load_split_support():
    """The Docling internals a split conversion needs, or None if unavailable.

    Splitting a document means converting page ranges separately and putting
    them back together, which needs three things Docling does not expose as a
    single supported entry point:

    * ``DoclingDocument.concatenate`` to merge the parts. It renumbers pages,
      re-indexes every ``self_ref`` and rewrites caption/reference/footnote
      links, so picture indices come out globally consistent.
    * ``HeadingHierarchyModel.assign_heading_levels``, documented as reusable
      outside the pipeline, to assign heading levels once over the merged
      document. Doing it per part instead is not equivalent: the numbering and
      font-style signals rank a heading against the rest of the document, and
      on a page range they see a different population. Measured on a 280-page
      manual: per-part inference reproduced 1131 of 2460 levels, and the merged
      pass reproduces all 2460.
    * ``extract_outline_from_pdfium`` to read the bookmarks. The pipeline only
      reads them when heading inference is enabled, which it no longer is here.

    All three are internal API. If a Docling upgrade moves them this returns
    None, the builder converts every document whole with Docling's own heading
    stage, and the only thing lost is the splitting.
    """
    try:
        import pypdfium2
        from docling.models.stages.heading_hierarchy.heading_hierarchy_model import (
            HeadingHierarchyModel,
        )
        from docling.utils.pdf_outline import extract_outline_from_pdfium
        from docling_core.types.doc.document import DoclingDocument

        if not hasattr(DoclingDocument, "concatenate"):
            raise ImportError("DoclingDocument.concatenate is gone")
        if not hasattr(HeadingHierarchyModel, "assign_heading_levels"):
            raise ImportError("HeadingHierarchyModel.assign_heading_levels is gone")
    except ImportError as e:
        logger.warning(
            f"Docling does not expose what a split conversion needs ({e}); "
            "documents will be converted whole."
        )
        return None

    return SimpleNamespace(
        concatenate=DoclingDocument.concatenate,
        heading_model_cls=HeadingHierarchyModel,
        read_outline=lambda path: extract_outline_from_pdfium(
            pypdfium2.PdfDocument(str(path))
        ),
    )


_SPLIT_SUPPORT = _load_split_support()


def plan_parts(page_count: int, split_pages: int) -> list[tuple[int, int]]:
    """Page ranges to convert a document in, as inclusive 1-based bounds.

    A single range means "convert it whole" and is the path a short document
    takes. Longer documents are cut into equal parts rather than into
    ``split_pages``-sized ones with a short remainder, so no worker is left
    holding a sliver while the others carry full parts.
    """
    pages = max(1, page_count)
    if split_pages <= 0 or pages <= split_pages:
        return [(1, pages)]
    count = math.ceil(pages / split_pages)
    # Spread the remainder one page at a time rather than letting it pile up in
    # the last part: ceil(2900/12) = 242 would leave a 238-page tail, and the
    # worker holding it finishes early while the others carry full parts.
    base, remainder = divmod(pages, count)
    ranges: list[tuple[int, int]] = []
    start = 1
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size - 1))
        start += size
    return ranges


def _init_docling_worker(num_threads: int, gpu_slot=None):
    """Process pool initializer: loads the Docling models once per worker."""
    global _converter, _gpu_slot
    _gpu_slot = gpu_slot
    logger.info(f"[Docling-{os.getpid()}] Initializing converter...")
    _converter = _create_converter(num_threads)
    if settings.PICTURE_DESCRIPTION_URL:
        logger.info(
            f"[Docling-{os.getpid()}] Figure descriptions enabled via "
            f"{settings.PICTURE_DESCRIPTION_URL} "
            f"(model={settings.PICTURE_DESCRIPTION_MODEL or '(unset)'})."
        )
    else:
        logger.info(f"[Docling-{os.getpid()}] Figure descriptions disabled.")
    logger.info(f"[Docling-{os.getpid()}] Ready.")


def _extract_figures(doc) -> list[dict]:
    """
    Pulls the rendered pictures out of a document as PNG bytes.

    The images are removed from the document afterwards: a PIL image travelling
    inside the pickled DoclingDocument blows the worker-to-parent transfer up
    (two figures grew a 6 KB pickle to 1.5 MB), while the PNG bytes returned
    here are handed to the parent as plain dicts.
    """
    figures: list[dict] = []

    for index, picture in enumerate(doc.pictures):
        try:
            image = picture.get_image(doc)
        except Exception as e:  # noqa: BLE001 - one bad crop must not fail a PDF
            logger.warning(f"Could not render picture {index}: {e}")
            image = None

        prov = picture.prov[0] if getattr(picture, "prov", None) else None
        if image is None or prov is None:
            continue

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        bbox = prov.bbox
        figures.append(
            {
                "picture_index": index,
                "page": prov.page_no,
                "bbox": (bbox.l, bbox.b, bbox.r, bbox.t),
                "width": image.width,
                "height": image.height,
                "png": buf.getvalue(),
            }
        )

    # Drop the images from every picture, including the ones skipped above.
    for picture in doc.pictures:
        picture.image = None

    return figures


def _count_missing_descriptions(doc) -> tuple[int, int]:
    """Counts pictures whose Docling-generated description is empty.

    Returns ``(missing, total)``.
    """
    total = len(doc.pictures)
    missing = sum(1 for p in doc.pictures if not _picture_description(p))
    return missing, total


def _release_device_cache() -> None:
    """Returns a worker's cached GPU blocks to the device.

    Torch's caching allocator keeps the peak a process reached in its own
    pool, so a Docling worker between parts goes on holding whatever its
    heaviest page needed. Measured on a live build, the three workers sat at
    6104 / 5588 / 4810 MB for twenty minutes without moving, rising when a
    heavier part came along and never falling; the parent's embedding then had
    4.8 GB to fit into 1.4 GB of headroom and did not.

    DOCLING_GPU_SLOTS cannot help with this: it rations who may *use* the
    device, not who may *hold* it. The same release was already added to the
    embedder; this is its other half.

    Never raises -- a worker that cannot free its cache has still converted
    its pages.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - releasing is best effort
        logger.debug(f"Could not release the device cache: {e}")


def _trim_heap() -> None:
    """
    Hands the heap freed by a conversion back to the operating system.

    Docling releases each page as it leaves the pipeline, but glibc keeps the
    freed blocks in its per-thread arenas, so a worker's RSS only ever climbs.
    Converting a 352-page manual twice left 6.7 GB resident; with this call
    between documents it stays at 2.7 GB, and the growth across the two passes
    drops from 1.4 GB to 83 MB. The call itself costs ~0.12 s against a ~250 s
    conversion.

    malloc_trim is glibc-only, so a missing symbol (musl, macOS) is not an
    error -- there is simply nothing to return.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # noqa: S110 - best-effort by design
        pass


def assign_heading_levels(doc, parsed_pages: dict, outline) -> None:
    """Assigns SectionHeaderItem.level over a finished document.

    Runs once per document, whether it was converted whole or as parts, so the
    numbering and font-style signals always rank a heading against the whole
    document's headings. Never raises: a document with unlevelled headings is
    worth more than no document.
    """
    if _SPLIT_SUPPORT is None:
        return
    try:
        model = _SPLIT_SUPPORT.heading_model_cls(
            HeadingHierarchyOptions(enabled=True)
        )
        model.assign_heading_levels(doc, parsed_pages=parsed_pages, outline=outline)
    except Exception as e:  # noqa: BLE001 - levels are an enrichment
        logger.error(f"Could not assign heading levels: {e}", exc_info=True)


def _save_markdown(doc, rel_path: str, pdf_path: Path) -> None:
    """Writes the markdown dump. A convenience artifact, never fatal."""
    try:
        md_path = settings.MARKDOWN_OUTPUT_DIR / Path(rel_path).with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        # A relative artifacts_dir makes Docling write relative image links, so
        # the dump stays valid when the tree is moved; the directory itself is
        # still created next to the markdown file.
        doc.save_as_markdown(
            md_path,
            artifacts_dir=Path(f"{md_path.stem}_artifacts"),
            image_mode=ImageRefMode.REFERENCED,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to save markdown for {pdf_path}: {e}")


def _convert_part_task(
    pdf_path: Path,
    rel_path: str,
    page_range: tuple[int, int] | None,
    save_markdown: bool,
    part_index: int = 1,
    part_count: int = 1,
):
    """
    Converts one document, or one page range of it, with the worker's converter.

    Returns ``(start_page, document, figures, parsed_pages)``. The document is
    stripped of its picture images and those come back as PNG records instead:
    a PIL image travelling inside the pickled DoclingDocument blows the
    worker-to-parent transfer up.

    ``page_range`` is None for a document short enough to convert in one go. In
    that case the worker also assigns the heading levels itself and returns no
    parsed pages, so nothing extra crosses the process boundary and the parent
    has nothing left to do. A page range instead returns its parsed pages, which
    the parent needs to level the merged document; measured at 0.17 MB per page
    once pickled, against 1.09 MB per page resident.

    Conversion errors are left to propagate through the future.
    """
    start_page = page_range[0] if page_range else 1
    label = f"{pdf_path.name}" + (
        f" pages {page_range[0]}-{page_range[1]}" if page_range else ""
    )
    logger.info(f"[Docling-{os.getpid()}] Converting {label}...")
    progress.emit_file(
        pdf_path,
        progress.STAGE_CONVERTING,
        worker=os.getpid(),
        part=start_page,
        part_index=part_index,
        part_count=part_count,
    )
    _begin_document(progress.relative_path(pdf_path), part=start_page)
    try:
        kwargs = {"page_range": page_range} if page_range else {}
        with gpu_slot(
            f"converting {label}",
            role=progress.ROLE_CONVERT,
            path=progress.relative_path(pdf_path),
            part_index=part_index,
            part_count=part_count,
            pages=(page_range[1] - page_range[0] + 1) if page_range else None,
        ):
            result = _converter.convert(str(pdf_path), **kwargs)
            # Inside the slot, like the embedder's. Handing the slot back
            # first leaves this worker holding its peak while whoever takes
            # it starts allocating -- which is the collision the slot exists
            # to prevent, moved a few lines later.
            _release_device_cache()
    finally:
        _end_document()
    doc = result.document

    parsed_pages = {
        page.page_no: page.parsed_page
        for page in result.pages
        if page.parsed_page is not None
    }

    if page_range is None:
        # Nothing to merge, so level it here and keep the parsed pages local.
        assign_heading_levels(doc, parsed_pages, _read_outline(pdf_path))
        parsed_pages = {}
        if save_markdown:
            # Written before the images are stripped, so the PNG files next to
            # it can actually be exported. A split document's dump is written
            # by the parent instead, from the merged document.
            _save_markdown(doc, rel_path, pdf_path)

    if settings.PICTURE_DESCRIPTION_URL:
        missing, total = _count_missing_descriptions(doc)
        if missing:
            logger.warning(
                f"{missing} of {total} figure(s) in {label} got no "
                "description (is the vision API at "
                f"{settings.PICTURE_DESCRIPTION_URL} running?)"
            )
        else:
            logger.info(f"{total} figure(s) in {label} got a description.")

    figures = _extract_figures(doc)

    # The per-page objects (parsed cells, rendered images, parser backends) are
    # dead once the figures are out, but only this worker will ever reuse the
    # memory they occupied, and it converts one document after another. Drop
    # the conversion result explicitly, then return the arenas to the OS.
    del result
    # Host arenas only: the device cache was already returned inside the slot,
    # while this worker still held the right to be using it.
    _trim_heap()

    progress.emit_file(
        pdf_path, progress.STAGE_CONVERTED, figures=len(figures), part=start_page
    )
    return start_page, doc, figures, parsed_pages


def _read_outline(pdf_path: Path):
    """The PDF's bookmarks, or None. Never raises: they are one signal of three."""
    if _SPLIT_SUPPORT is None:
        return None
    try:
        return _SPLIT_SUPPORT.read_outline(pdf_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read the outline of {pdf_path}: {e}")
        return None


def merge_parts(parts: list[tuple[int, object, list[dict]]]):
    """Merges converted page ranges back into one document and its figures.

    ``parts`` is ``(start_page, document, figures)`` in any order. Figure
    indices are part-local, and ``concatenate`` re-indexes pictures across the
    merged document in part order, so each part's figures are shifted by the
    number of pictures in the parts before it.
    """
    ordered = sorted(parts, key=lambda part: part[0])
    merged = _SPLIT_SUPPORT.concatenate([doc for _, doc, _ in ordered])

    figures: list[dict] = []
    offset = 0
    for _, doc, part_figures in ordered:
        for figure in part_figures:
            shifted = dict(figure)
            shifted["picture_index"] = figure["picture_index"] + offset
            figures.append(shifted)
        offset += len(doc.pictures)
    return merged, figures


def _ingest_document(
    session: Session,
    collection,
    embedder,
    manual_id: str,
    pdf_path: Path,
    pdf_root: Path,
    doc,
    figures: list[dict],
) -> int:
    """
    Chunks a converted document, embeds it and stores it in ChromaDB.

    The rendered figures are written to the SQLite ``figures`` table and the
    chunk that describes each of them only carries the resulting ``figure_id``
    into Chroma, so the image bytes never leave the relational database.

    Runs in the main process while the worker processes keep converting, so the
    GPU serves the Docling pipelines and the embedding model at the same time.
    Returns the number of chunks written.
    """
    manual = session.get(Manual, manual_id)
    if not manual:
        logger.error(f"Manual {manual_id} not found in DB. Skipping.")
        return 0

    chunks = chunk_document(doc, manual)
    try:
        rel_path = pdf_path.relative_to(pdf_root)
    except ValueError:
        rel_path = Path(pdf_path.name)

    logger.info(f"Generated {len(chunks)} chunks for {rel_path}")
    if not chunks:
        # A document that yields nothing is still converted; without the mark
        # every later build would convert it again.
        manual.converted_at = datetime.now(UTC)
        session.commit()
        return 0

    figures_by_index = {f["picture_index"]: f for f in figures}
    # One row per picture, whatever the chunking does. A long figure caption is
    # split across several chunks, and every one of them carries the same
    # picture_index: creating the row inside the loop would store the PNG once
    # per chunk and hand each of them a different figure_id.
    figure_rows: dict[int, Figure] = {}

    ids = [f"{manual.id}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = []
    stored_figures = 0

    for i, c in enumerate(chunks):
        chunk_meta = c["metadata"]
        meta = {
            "source": str(rel_path),
            "manual_id": str(manual.id),
            "chunk_index": float(i),
        }

        if chunk_meta.get("bookmark_id"):
            meta["bookmark_id"] = str(chunk_meta["bookmark_id"])

        # Chroma metadata values must be str/int/float/bool
        if chunk_meta.get("type"):
            meta["type"] = str(chunk_meta["type"])
        if chunk_meta.get("page") is not None:
            meta["page"] = int(chunk_meta["page"])

        if chunk_meta.get("type") == "figure":
            index = chunk_meta.get("picture_index")
            record = figures_by_index.get(index)
            existing = figure_rows.get(index)
            if existing is not None:
                meta["figure_id"] = existing.id
            elif record is None:
                logger.warning(
                    f"No rendered image for picture {index} of {rel_path}: "
                    "the chunk is stored without a figure reference."
                )
            else:
                bbox = record["bbox"]
                figure = Figure(
                    id=str(uuid.uuid4()),
                    manual_id=str(manual.id),
                    bookmark_id=chunk_meta.get("bookmark_id"),
                    picture_index=int(record["picture_index"]),
                    page=int(record["page"]),
                    bbox_l=float(bbox[0]),
                    bbox_b=float(bbox[1]),
                    bbox_r=float(bbox[2]),
                    bbox_t=float(bbox[3]),
                    caption=chunk_meta.get("figure_caption"),
                    labels=chunk_meta.get("figure_labels"),
                    description=chunk_meta.get("figure_description"),
                    mime_type="image/png",
                    width=record["width"],
                    height=record["height"],
                    image=record["png"],
                )
                session.add(figure)
                stored_figures += 1
                figure_rows[index] = figure
                meta["figure_id"] = figure.id

        metadatas.append(meta)

    # Commit the figures before touching Chroma: a figure row without its chunk
    # is harmless, a chunk pointing at a missing figure id is not.
    if stored_figures:
        session.commit()
        logger.info(f"Stored {stored_figures} figure(s) in SQLite for {rel_path}")

    # One of the GPU slots, so embedding a finished document costs a
    # converting worker for its duration instead of colliding with three.
    with gpu_slot(
        f"embedding {len(documents)} chunks of {rel_path}",
        role=progress.ROLE_EMBED,
        path=str(rel_path),
        chunks=len(documents),
    ):
        # Both the move and the release happen inside the slot. A slot handed
        # back while the embedder still holds the device frees the right to
        # run but not the memory to run in, and releasing after the slot is
        # gone would race the worker that took it.
        with embedder.on_device():
            embeddings = embedder.embed_documents(documents)

    for start in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
        end = start + CHROMA_ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    logger.info(f"Added {len(chunks)} chunks to ChromaDB for {pdf_path.name}")

    # Only now is the manual really in the database. Until this commit lands,
    # its row is just the metadata pass's promise that the file exists.
    manual.converted_at = datetime.now(UTC)
    session.commit()
    return len(chunks)


def select_pdf_files(pdf_dir: Path, include: list[str] | None = None) -> list[Path]:
    """
    Lists the PDFs under ``pdf_dir``, optionally narrowed by glob patterns.

    Patterns are matched against each file's path relative to ``pdf_dir``, in
    POSIX form, case-sensitively, and a file is kept when it matches any of
    them. These are :mod:`fnmatch` patterns, so ``*`` also matches ``/``:
    ``zOS/V3R1/*`` selects that directory and everything below it.

    Narrowing the scan this way rather than by pointing ``--pdf_dir`` at the
    subdirectory matters, because ``pdf_dir`` is also the anchor every stored
    ``relative_path`` is computed from. Keeping the anchor at the corpus root
    means a subset build writes the same paths a whole-corpus build would, so
    subsets can be built one at a time and still add up to one consistent
    database.
    """
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not include:
        return pdf_files

    selected = []
    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(pdf_dir).as_posix()
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in include):
            selected.append(pdf_path)
    return selected


def build(
    pdf_dir: Path,
    reset: bool,
    save_markdown: bool = False,
    include: list[str] | None = None,
    progress_file: Path | None = None,
    min_pages: int | None = None,
    max_pages: int | None = None,
):
    check_dependencies()

    # Every process in the run appends to this file; truncate it first so a
    # monitor sees this build and not the tail of the previous one. Configure
    # before the pools are created: they inherit the setting through the
    # environment, which is what survives the "spawn" start method.
    if progress_file is not None:
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            progress_file.write_text("", encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not open progress file {progress_file}: {e}")
            progress_file = None
    progress.configure(progress_file, pdf_dir)

    # Prepare directories
    if reset:
        if settings.DB_FILE_PATH.exists():
            logger.warning(f"Deleting existing DB: {settings.DB_FILE_PATH}")
            settings.DB_FILE_PATH.unlink()

        if settings.CHROMADB_PATH.exists():
            logger.warning(f"Resetting ChromaDB directory: {settings.CHROMADB_PATH}")
            shutil.rmtree(settings.CHROMADB_PATH)

        if settings.MARKDOWN_OUTPUT_DIR.exists():
            logger.warning(
                f"Resetting Markdown Output directory: {settings.MARKDOWN_OUTPUT_DIR}"
            )
            shutil.rmtree(settings.MARKDOWN_OUTPUT_DIR)

    if save_markdown:
        settings.MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize DB
    logger.info(f"Initializing Database at {settings.DB_FILE_PATH}...")
    if not settings.DB_FILE_PATH.parent.exists():
        settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()

    # Process PDFs - Recursive scan
    pdf_files = select_pdf_files(pdf_dir, include)
    if not pdf_files:
        if include:
            logger.warning(
                f"No PDF files under {pdf_dir} matched {include}"
            )
        else:
            logger.warning(f"No PDF files found in {pdf_dir}")
        progress.emit(
            "run_start",
            pdf_dir=str(pdf_dir),
            include=list(include) if include else None,
            reset=reset,
            total=0,
        )
        progress.emit("run_end", found=0, skipped=0, converted=0, failed=0, chunks=0)
        return

    if include:
        logger.info(f"Found {len(pdf_files)} PDF file(s) matching {include}.")
    else:
        logger.info(f"Found {len(pdf_files)} PDF files.")

    # Sort files by size (descending) to process largest files first (LPT scheduling)
    # This helps balanced load distribution among workers and reduces total make-span
    sizes = {}
    for pdf_path in pdf_files:
        try:
            sizes[pdf_path] = pdf_path.stat().st_size
        except OSError:
            sizes[pdf_path] = 0
    pdf_files.sort(key=lambda p: sizes[p], reverse=True)

    progress.emit(
        "run_start",
        pdf_dir=str(pdf_dir),
        include=list(include) if include else None,
        reset=reset,
        total=len(pdf_files),
        workers=max(1, settings.DOCLING_WORKERS),
        metadata_workers=max(1, settings.METADATA_WORKERS),
        min_pages=min_pages,
        max_pages=max_pages,
        gpu_slots=settings.DOCLING_GPU_SLOTS or None,
    )
    # One event per file rather than one list: a line has to stay small enough
    # for the kernel to append it atomically next to the workers' writes.
    for pdf_path in pdf_files:
        progress.emit(
            "discovered",
            path=progress.relative_path(pdf_path),
            size=sizes[pdf_path],
        )

    # Phase 1: hash + pypdf metadata extraction (CPU bound, no GPU involved)
    # The task itself lives in pdf_utils: spawned workers import the module that
    # defines it, so keeping it out of builder.py means they load only pypdf
    # instead of the whole Docling/torch stack.
    metadata_workers = max(1, settings.METADATA_WORKERS)
    if metadata_workers <= 1:
        logger.info("Extracting PDF metadata inline (METADATA_WORKERS <= 1).")
        results = list(map(extract_pdf_fingerprint, pdf_files))
    else:
        logger.info(f"Extracting PDF metadata with {metadata_workers} processes.")
        with _make_process_executor(metadata_workers) as executor:
            # executor.map preserves the input order
            results = list(executor.map(extract_pdf_fingerprint, pdf_files))

    # Phase 2: relational DB sync (main process only, SQLite has one writer)
    tasks: list[tuple[Path, str, str, int]] = []
    stale_manual_ids: list[str] = []
    skipped = 0
    failed = 0
    converted = 0
    deferred = 0
    total_chunks = 0

    session = SessionLocal()
    try:
        for result in results:
            pdf_path = result["pdf_path"]
            if result["error"] or result["metadata"] is None:
                logger.error(f"Failed to read {pdf_path}: {result['error']}")
                progress.emit_file(
                    pdf_path, progress.STAGE_FAILED, error=str(result["error"])
                )
                failed += 1
                continue

            try:
                manual, updated, existed = sync_manual_to_db(
                    session,
                    pdf_path,
                    pdf_dir,
                    result["file_hash"],
                    result["metadata"],
                )
            except Exception as e:
                logger.error(f"Failed to sync DB for {pdf_path}: {e}", exc_info=True)
                progress.emit_file(
                    pdf_path, progress.STAGE_FAILED, error=f"{type(e).__name__}: {e}"
                )
                failed += 1
                continue

            try:
                rel_path_str = str(pdf_path.relative_to(pdf_dir))
            except ValueError:
                rel_path_str = pdf_path.name

            pages = manual.page_count or 0
            if (min_pages is not None and pages < min_pages) or (
                max_pages is not None and pages > max_pages
            ):
                logger.info(
                    f"File {rel_path_str} ({pages} pages) is outside the "
                    "requested page range. Leaving it for another run."
                )
                deferred += 1
                continue

            unchanged = not updated and not reset
            if unchanged and manual.converted_at is not None:
                logger.info(f"File {rel_path_str} unchanged. Skipping.")
                progress.emit_file(
                    pdf_path, progress.STAGE_SKIPPED, pages=manual.page_count
                )
                skipped += 1
                continue

            if unchanged:
                # The row is here but the conversion never finished: an
                # interrupted build, or one whose ingestion raised. Convert it
                # again rather than trusting the hash, which only says the file
                # on disk has not changed since the metadata pass wrote the row.
                logger.info(
                    f"File {rel_path_str} was registered but never converted. "
                    "Resuming it."
                )

            progress.emit_file(
                pdf_path, progress.STAGE_QUEUED, pages=manual.page_count
            )
            tasks.append(
                (pdf_path, str(manual.id), rel_path_str, manual.page_count or 0)
            )

            # A manual that already existed still holds whatever chunks a
            # previous run wrote; they must be dropped before re-inserting.
            if existed:
                stale_manual_ids.append(str(manual.id))

        logger.info(f"Queued {len(tasks)} files for processing.")

        if not tasks:
            logger.info("No files to process.")
            return

        # Initialize Chroma components only when there is work to do, so the
        # embedding model is not loaded for a no-op build.
        logger.info(f"Initializing ChromaDB at {settings.CHROMADB_PATH}...")
        client = chromadb.PersistentClient(path=str(settings.CHROMADB_PATH))
        embedder = get_embedder()
        if embedder is None:
            sys.exit(1)

        # embedding_function=None: every vector is computed here and passed
        # explicitly, so Chroma must never embed anything on its own.
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata=collection_metadata(embedder),
        )

        # get_or_create_collection ignores the metadata of an existing
        # collection, so an older collection keeps its own model name: refuse to
        # mix vectors from two models. After a reset the collection is fresh.
        if not reset:
            try:
                check_collection_model(collection, embedder.model_name)
            except RuntimeError as e:
                logger.error(str(e))
                sys.exit(1)

        for mid in stale_manual_ids:
            try:
                logger.info(f"Deleting stale chunks for manual {mid}.")
                collection.delete(where={"manual_id": mid})
            except Exception as e:
                logger.error(f"Failed to delete stale chunks for manual {mid}: {e}")

        # Phase 3: Docling conversion in worker processes, ingestion in the parent
        num_workers = max(1, settings.DOCLING_WORKERS)
        threads_per_worker = max(1, settings.DOCLING_NUM_THREADS // num_workers)
        logger.info(
            f"Starting {num_workers} Docling worker process(es) with "
            f"{threads_per_worker} thread(s) each."
        )

        # Docling workers and this process's embedding model share the GPU.
        # The slots are what stop them from racing to exhaust it; the parent
        # takes one to embed, which simply costs a converting worker until the
        # embedding is done.
        global _gpu_slot
        slots = settings.DOCLING_GPU_SLOTS
        _gpu_slot = mp.get_context("spawn").Semaphore(slots) if slots > 0 else None
        if _gpu_slot is not None:
            logger.info(
                f"Limiting the GPU to {slots} concurrent user(s): "
                f"{num_workers} Docling worker(s) plus this process's embedder."
            )

        with _make_process_executor(
            num_workers,
            initializer=_init_docling_worker,
            initargs=(threads_per_worker, _gpu_slot),
        ) as executor:
            # A document is submitted as one task per page range. Every part
            # is the same size, so a worker's peak memory follows
            # DOCLING_SPLIT_PAGES rather than the longest document in the
            # corpus -- and a 2900-page manual no longer occupies one worker
            # while the rest of the pool waits for it.
            split_pages = settings.DOCLING_SPLIT_PAGES if _SPLIT_SUPPORT else 0
            futures: dict[Future, tuple[Path, str, str]] = {}
            pending: dict[str, dict] = {}
            for pdf_path, manual_id, rel_path_str, page_count in tasks:
                ranges = plan_parts(page_count, split_pages)
                single = len(ranges) == 1
                pending[manual_id] = {
                    "pdf_path": pdf_path,
                    "expected": len(ranges),
                    "parts": [],
                    "parsed_pages": {},
                    "failed": False,
                }
                if not single:
                    logger.info(
                        f"{rel_path_str}: {page_count} pages, converting as "
                        f"{len(ranges)} parts of ~{ranges[0][1]} pages."
                    )
                progress.emit_file(
                    pdf_path, progress.STAGE_QUEUED, parts=len(ranges)
                )
                for index, page_range in enumerate(ranges, start=1):
                    future = executor.submit(
                        _convert_part_task,
                        pdf_path,
                        rel_path_str,
                        None if single else page_range,
                        save_markdown,
                        index,
                        len(ranges),
                    )
                    futures[future] = (pdf_path, manual_id, rel_path_str)

            # as_completed takes its own snapshot, so removing entries here is
            # safe -- and necessary. A Future holds on to its result, so every
            # part's document, figure PNGs and parsed pages would stay resident
            # in the parent for the whole build: measured at +4.4 GB after five
            # documents (3942 pages), which is 1.1 MB per page and matches the
            # parsed pages exactly. Over 172k pages that is not survivable.
            for future in as_completed(list(futures)):
                pdf_path, manual_id, rel_path_str = futures.pop(future)
                state = pending[manual_id]
                try:
                    start_page, part_doc, part_figures, parsed = future.result()
                except BrokenProcessPool as e:
                    logger.error(
                        "A Docling worker process died (out of memory / CUDA "
                        "error?) - lower DOCLING_WORKERS or DOCLING_SPLIT_PAGES "
                        "and retry."
                    )
                    raise RuntimeError("Docling worker process pool broke") from e
                except Exception as e:
                    logger.error(
                        f"Failed to convert {pdf_path}: {e}", exc_info=True
                    )
                    if not state["failed"]:
                        # One bad part fails the document once, not once per
                        # part, and the parts already in hand are dropped.
                        state["failed"] = True
                        state["parts"] = []
                        state["parsed_pages"] = {}
                        progress.emit_file(
                            pdf_path,
                            progress.STAGE_FAILED,
                            error=f"{type(e).__name__}: {e}",
                        )
                        failed += 1
                    continue

                if state["failed"]:
                    continue
                state["parts"].append((start_page, part_doc, part_figures))
                state["parsed_pages"].update(parsed)
                if len(state["parts"]) < state["expected"]:
                    continue

                if state["expected"] == 1:
                    _, doc, figures = state["parts"][0]
                else:
                    doc, figures = merge_parts(state["parts"])
                    assign_heading_levels(
                        doc, state["parsed_pages"], _read_outline(pdf_path)
                    )
                    if save_markdown:
                        # The parts were dumped as nothing; the merged document
                        # is the useful artifact. Its pictures live in SQLite,
                        # so the dump carries no image files.
                        _save_markdown(doc, rel_path_str, pdf_path)
                pending.pop(manual_id, None)

                # Nothing below needs the raw parts, and holding them across
                # the embedding step doubles what the parent carries.
                state["parts"] = []
                state["parsed_pages"] = {}

                progress.emit_file(
                    pdf_path, progress.STAGE_INGESTING, figures=len(figures)
                )
                try:
                    chunks = _ingest_document(
                        session,
                        collection,
                        embedder,
                        manual_id,
                        pdf_path,
                        pdf_dir,
                        doc,
                        figures,
                    )
                    total_chunks += chunks
                    progress.emit_file(
                        pdf_path,
                        progress.STAGE_DONE,
                        chunks=chunks,
                        figures=len(figures),
                    )
                    converted += 1
                except Exception as e:
                    logger.error(
                        f"Failed to ingest {pdf_path}: {e}", exc_info=True
                    )
                    progress.emit_file(
                        pdf_path,
                        progress.STAGE_FAILED,
                        error=f"{type(e).__name__}: {e}",
                    )
                    failed += 1
    finally:
        session.close()
        # In the finally so that an early return, a sys.exit() or a broken
        # worker pool still closes the run out for whoever is watching.
        progress.emit(
            "run_end",
            found=len(pdf_files),
            skipped=skipped,
            converted=converted,
            failed=failed,
            deferred=deferred,
            chunks=total_chunks,
        )

    logger.info(
        f"Summary: {len(pdf_files)} file(s) found, {skipped} unchanged, "
        f"{deferred} outside the page range, {converted} converted, "
        f"{failed} failed, {total_chunks} chunk(s) stored."
    )
    logger.info("Build complete.")
    logger.info(f"Database saved to {settings.CHROMADB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB from PDFs using Docling."
    )
    parser.add_argument(
        "--pdf_dir", type=str, required=True, help="Directory containing PDF files."
    )
    parser.add_argument(
        "--save-markdown", action="store_true", help="Save intermediate Markdown files."
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete output directory before starting."
    )
    parser.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help=(
            "Only convert PDFs whose path relative to --pdf_dir matches this "
            "glob (repeatable; '*' also matches '/')."
        ),
    )
    parser.add_argument("--min-pages", type=int, metavar="N")
    parser.add_argument("--max-pages", type=int, metavar="N")

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    build(
        pdf_dir,
        args.reset,
        args.save_markdown,
        args.include,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
    )
