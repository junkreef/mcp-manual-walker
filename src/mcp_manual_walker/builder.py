import argparse
import ctypes
import io
import logging
import multiprocessing as mp
import os
import shutil
import sys
import uuid
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# Imports for dependencies
from sqlalchemy import select
from sqlalchemy.orm import Session

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
    from docling_core.types.doc import ImageRefMode
    # We might need specific options if we want to speed up or customize
except ImportError as e:
    logger.error(f"Failed to import docling: {e}", exc_info=True)
    DocumentConverter = None

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

    # Derive section-header levels from PDF bookmarks / numbering / font style
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    # The font-style signal reads the parsed PDF cells, which the pipeline
    # discards unless they are explicitly kept.
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
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }
    )


def _init_docling_worker(num_threads: int):
    """Process pool initializer: loads the Docling models once per worker."""
    global _converter
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


def _convert_pdf_task(pdf_path: Path, rel_path: str, save_markdown: bool):
    """
    Converts a single PDF with the worker-local converter.

    Returns ``(document, figures)``: the DoclingDocument stripped of its picture
    images, plus those images as plain PNG records. Both are pickled back to the
    parent process, which is far cheaper than re-running Docling there.
    Conversion errors are left to propagate through the future.
    """
    logger.info(f"[Docling-{os.getpid()}] Converting {pdf_path.name}...")
    result = _converter.convert(str(pdf_path))
    doc = result.document

    if save_markdown:
        # A markdown dump is a convenience artifact; never fail the conversion
        # because it could not be written. It is written before the images are
        # stripped, so the PNG files next to it can actually be exported.
        try:
            md_path = settings.MARKDOWN_OUTPUT_DIR / Path(rel_path).with_suffix(".md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            # A relative artifacts_dir makes Docling write relative image
            # links, so the dump stays valid when the tree is moved; the
            # directory itself is still created next to the markdown file.
            doc.save_as_markdown(
                md_path,
                artifacts_dir=Path(f"{md_path.stem}_artifacts"),
                image_mode=ImageRefMode.REFERENCED,
            )
        except Exception as e:
            logger.error(f"Failed to save markdown for {pdf_path}: {e}")

    if settings.PICTURE_DESCRIPTION_URL:
        missing, total = _count_missing_descriptions(doc)
        if missing:
            logger.warning(
                f"{missing} of {total} figure(s) in {rel_path} got no "
                "description (is the vision API at "
                f"{settings.PICTURE_DESCRIPTION_URL} running?)"
            )
        else:
            logger.info(f"{total} figure(s) in {rel_path} got a description.")

    figures = _extract_figures(doc)

    # The per-page objects (parsed cells, rendered images, parser backends) are
    # dead once the figures are out, but only this worker will ever reuse the
    # memory they occupied, and it converts one document after another. Drop
    # the conversion result explicitly, then return the arenas to the OS.
    del result
    _trim_heap()

    return doc, figures


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
        return 0

    figures_by_index = {f["picture_index"]: f for f in figures}

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
            if record is None:
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
                meta["figure_id"] = figure.id

        metadatas.append(meta)

    # Commit the figures before touching Chroma: a figure row without its chunk
    # is harmless, a chunk pointing at a missing figure id is not.
    if stored_figures:
        session.commit()
        logger.info(f"Stored {stored_figures} figure(s) in SQLite for {rel_path}")

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
    return len(chunks)


def build(pdf_dir: Path, reset: bool, save_markdown: bool = False):
    check_dependencies()

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
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files.")

    # Sort files by size (descending) to process largest files first (LPT scheduling)
    # This helps balanced load distribution among workers and reduces total make-span
    pdf_files.sort(key=lambda p: p.stat().st_size, reverse=True)

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
    tasks: list[tuple[Path, str, str]] = []
    stale_manual_ids: list[str] = []
    skipped = 0
    failed = 0

    session = SessionLocal()
    try:
        for result in results:
            pdf_path = result["pdf_path"]
            if result["error"] or result["metadata"] is None:
                logger.error(f"Failed to read {pdf_path}: {result['error']}")
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
                failed += 1
                continue

            try:
                rel_path_str = str(pdf_path.relative_to(pdf_dir))
            except ValueError:
                rel_path_str = pdf_path.name

            if not updated and not reset:
                logger.info(f"File {rel_path_str} unchanged. Skipping.")
                skipped += 1
                continue

            tasks.append((pdf_path, str(manual.id), rel_path_str))

            # A manual that already existed and changed still holds its old
            # chunks in Chroma; they must be dropped before re-inserting.
            if existed and updated:
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

        converted = 0
        total_chunks = 0

        with _make_process_executor(
            num_workers,
            initializer=_init_docling_worker,
            initargs=(threads_per_worker,),
        ) as executor:
            futures: dict[Future, tuple[Path, str]] = {}
            for pdf_path, manual_id, rel_path_str in tasks:
                future = executor.submit(
                    _convert_pdf_task, pdf_path, rel_path_str, save_markdown
                )
                futures[future] = (pdf_path, manual_id)

            for future in as_completed(futures):
                pdf_path, manual_id = futures[future]
                try:
                    doc, figures = future.result()
                except BrokenProcessPool as e:
                    logger.error(
                        "A Docling worker process died (out of memory / CUDA "
                        "error?) - lower DOCLING_WORKERS and retry."
                    )
                    raise RuntimeError("Docling worker process pool broke") from e
                except Exception as e:
                    logger.error(
                        f"Failed to convert {pdf_path}: {e}", exc_info=True
                    )
                    failed += 1
                    continue

                try:
                    total_chunks += _ingest_document(
                        session,
                        collection,
                        embedder,
                        manual_id,
                        pdf_path,
                        pdf_dir,
                        doc,
                        figures,
                    )
                    converted += 1
                except Exception as e:
                    logger.error(
                        f"Failed to ingest {pdf_path}: {e}", exc_info=True
                    )
                    failed += 1
    finally:
        session.close()

    logger.info(
        f"Summary: {len(pdf_files)} file(s) found, {skipped} unchanged, "
        f"{converted} converted, {failed} failed, {total_chunks} chunk(s) stored."
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

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    build(pdf_dir, args.reset, args.save_markdown)
