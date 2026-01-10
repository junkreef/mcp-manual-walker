import argparse
import logging
import shutil
import sys
import uuid
import warnings
import queue
import threading
from pathlib import Path
from typing import Optional

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
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, AcceleratorOptions, AcceleratorDevice, RapidOcrOptions
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
    from mcp_manual_walker.chunking import chunk_text_by_coordinates
    from mcp_manual_walker.config import settings
    from mcp_manual_walker.database import SessionLocal, init_db
    from mcp_manual_walker.embeddings import get_embedding_function as get_ef
    from mcp_manual_walker.models import Bookmark, Manual
    from mcp_manual_walker.pdf_utils import calculate_file_hash, extract_pdf_metadata
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    sys.exit(1)


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


def get_embedding_function():
    # Helper to get the embedding function via factory
    ef = get_ef()
    if ef is None:
        logger.error(
            "Could not load any embedding function. Please install 'fastembed' or 'sentence-transformers'."
        )
        sys.exit(1)
    return ef


def sync_manual_to_db(
    session: Session, pdf_path: Path, pdf_root: Path
) -> tuple[Manual, bool]:
    """
    Syncs the Manual and Bookmarks to the SQLite DB.
    Returns:
        (Manual, bool): The manual object and a boolean indicating if it was updated/new (True) or unchanged (False).
    """
    # Use consistent relative path from the root PDF directory
    try:
        rel_path_str = str(pdf_path.relative_to(pdf_root))
    except ValueError:
        # Fallback if path is not relative to root (should be rare in current usage)
        rel_path_str = pdf_path.name

    file_hash = calculate_file_hash(pdf_path)

    # Extract metadata including bookmarks with TOP coordinates
    metadata = extract_pdf_metadata(pdf_path)
    if not metadata:
        raise ValueError(f"Failed to extract metadata from {pdf_path}")

    stmt = select(Manual).where(Manual.relative_path == str(pdf_path.relative_to(pdf_root)))
    manual = session.execute(stmt).scalars().first()

    if manual:
        logger.info(f"Manual {str(pdf_path.relative_to(pdf_root))} found in DB. Checking hash...")
        if manual.file_hash == file_hash:
            logger.info("Hash match. Skipping DB sync (bookmarks).")
            return manual, False
        else:
            logger.info("Hash mismatch. Updating...")
            # Delete old bookmarks
            manual.bookmarks.clear()
    else:
        logger.info(f"Creating new Manual entry for {str(pdf_path.relative_to(pdf_root))}")
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
    return manual, True


def docling_worker(pdf_queue: queue.Queue, doc_queue: queue.Queue, pdf_root: Path, worker_id: int):
    """
    Worker function for Docling conversion.
    Each worker initializes its own DocumentConverter to parallelize GPU usage.
    """
    logger.info(f"[Docling-{worker_id}] Initializing converter...")
    
    try:
        pipeline_options = ThreadedPdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
            num_threads=settings.DOCLING_NUM_THREADS
        )
        # Increase batch sizes to improve GPU utilization
        pipeline_options.ocr_batch_size = settings.DOCLING_OCR_BATCH_SIZE
        pipeline_options.layout_batch_size = settings.DOCLING_LAYOUT_BATCH_SIZE
        pipeline_options.table_batch_size = settings.DOCLING_TABLE_BATCH_SIZE
        

        pipeline_options.ocr_options = RapidOcrOptions(
            backend="torch",
        )   

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    except Exception as e:
        logger.error(f"[Docling-{worker_id}] Failed to initialize converter: {e}", exc_info=True)
        return

    logger.info(f"[Docling-{worker_id}] Ready.")

    while True:
        item = pdf_queue.get()
        if item is None:
            # Sentinel value to stop the worker
            pdf_queue.task_done()
            break

        pdf_path, manual_id = item
        
        try:
            logger.info(f"[Docling-{worker_id}] Converting {pdf_path.name}...")
            result = converter.convert(str(pdf_path))
            
            # Pass to Embedding Worker
            doc_queue.put((manual_id, pdf_path, result))
            logger.info(f"[Docling-{worker_id}] Queued {pdf_path.name} for embedding.")

        except Exception as e:
            logger.error(f"[Docling-{worker_id}] Failed to convert {pdf_path}: {e}", exc_info=True)
        finally:
            pdf_queue.task_done()
    
    logger.info(f"[Docling-{worker_id}] Finished.")


def embedding_worker(q: queue.Queue, collection, pdf_root: Path, save_markdown: bool):
    """
    Worker function to process documents from the queue:
    1. Chunking (CPU)
    2. Embedding (GPU/CPU)
    3. Insert into ChromaDB
    4. Save Markdown (optional)
    """
    logger.info("[Embedding-Worker] Started.")
    
    # Create DB session once per worker for better performance
    with SessionLocal() as session:
        while True:
            item = q.get()
            if item is None:
                # Sentinel value to stop the worker
                q.task_done()
                break

            manual_id, pdf_path, doc_result = item
            
            try:
                logger.info(f"[Embedding-Worker] Processing {pdf_path.name}...")

                # Save Markdown (optional)
                if save_markdown:
                    try:
                        md_content = doc_result.document.export_to_markdown()
                        rel_path = pdf_path.relative_to(pdf_root)
                        md_path = settings.MARKDOWN_OUTPUT_DIR / rel_path.with_suffix(".md")
                        md_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(md_content)
                    except Exception as e:
                        logger.error(f"[Embedding-Worker] Failed to save markdown for {pdf_path}: {e}")

                # Re-fetch manual object for chunking
                manual = session.get(Manual, manual_id)
                if not manual:
                    logger.error(f"[Embedding-Worker] Manual {manual_id} not found in DB. Skipping.")
                    continue

                # 3. Coordinate-Based Chunking
                chunks = chunk_text_by_coordinates(doc_result.document, manual)
                logger.info(f"[Embedding-Worker] Generated {len(chunks)} chunks for {pdf_path.relative_to(pdf_root)}")

                if not chunks:
                    continue

                # 4. Add to ChromaDB
                ids = [f"{manual.id}_{i}" for i in range(len(chunks))]
                documents = [c["text"] for c in chunks]
                metadatas = []

                rel_path = pdf_path.relative_to(pdf_root)

                for i, c in enumerate(chunks):
                    meta = {
                        "source": str(rel_path),
                        "manual_id": str(manual.id),
                        "chunk_index": float(i),
                    }

                    if c["metadata"].get("bookmark_id"):
                        meta["bookmark_id"] = str(c["metadata"]["bookmark_id"])

                    metadatas.append(meta)

                collection.add(ids=ids, documents=documents, metadatas=metadatas)
                logger.info(f"[Embedding-Worker] Added {len(chunks)} chunks to ChromaDB for {pdf_path.name}")

            except Exception as e:
                logger.error(f"[Embedding-Worker] Failed to process {pdf_path}: {e}", exc_info=True)
            finally:
                q.task_done()
    
    logger.info("[Embedding-Worker] Finished.")


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
    
    # Initialize Chroma components (Main thread)
    logger.info(f"Initializing ChromaDB at {settings.CHROMADB_PATH}...")
    client = chromadb.PersistentClient(path=str(settings.CHROMADB_PATH))
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name="manual_chunks",
        embedding_function=embedding_fn,
        metadata={"description": "Chunks from PDF manuals"},
    )

    # Process PDFs - Recursive scan
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files.")

    # Sort files by size (descending) to process largest files first (LPT scheduling)
    # This helps balanced load distribution among workers and reduces total make-span
    pdf_files.sort(key=lambda p: p.stat().st_size, reverse=True)

    # Queues
    pdf_queue = queue.Queue()
    doc_queue = queue.Queue(maxsize=10) # Limit intermediate queue

    # Pre-process DB Sync and fill Queue
    session = SessionLocal()
    try:
        count = 0
        for pdf_path in pdf_files:
            try:
                # 1. Sync to Relational DB (Main thread - fast)
                manual, updated = sync_manual_to_db(session, pdf_path, pdf_dir)

                if not updated and not reset:
                    logger.info(
                        f"File {str(pdf_path.relative_to(pdf_dir))} unchanged. Skipping."
                    )
                    continue
                
                pdf_queue.put((pdf_path, str(manual.id)))
                count += 1
            except Exception as e:
                logger.error(f"Failed to sync DB for {pdf_path}: {e}")
        
        logger.info(f"Queued {count} files for processing.")
        
    finally:
        session.close()

    if count == 0:
        logger.info("No files to process.")
        return

    # Start Workers
    
    # 1 Docling Worker (GPU) - Single thread to avoid GPU contention and context switching overhead
    # Previous tests showed that 3 workers caused 4x slowdown per file due to resource contention.
    docling_threads = []
    for i in range(1):
        t = threading.Thread(
            target=docling_worker,
            args=(pdf_queue, doc_queue, pdf_dir, i+1),
            daemon=True
        )
        t.start()
        docling_threads.append(t)

    # 1 Embedding Worker
    embedding_thread = threading.Thread(
        target=embedding_worker, 
        args=(doc_queue, collection, pdf_dir, save_markdown),
        daemon=True
    )
    embedding_thread.start()

    # Wait for PDF queue to be empty (Docling workers finished processing all items)
    pdf_queue.join()
    logger.info("All PDFs have been processed by Docling workers.")

    # Stop Docling workers
    for _ in range(1):
        pdf_queue.put(None)
    
    for t in docling_threads:
        t.join()

    # Wait for Docling results to be processed by Embedding worker
    doc_queue.join()
    logger.info("All documents have been embedded.")

    # Stop Embedding worker
    doc_queue.put(None)
    embedding_thread.join()

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
