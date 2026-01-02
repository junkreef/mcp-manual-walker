import argparse
import logging
import shutil
import sys
import uuid
import warnings
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
    from docling.document_converter import DocumentConverter
    # We might need specific options if we want to speed up or customize
except ImportError:
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
        raise Exception(f"Failed to extract metadata from {pdf_path}")

    # Check existence by file_hash or relative_path?
    # Duplicate filenames in different folders possible? Yes.
    # We need a stable identifier. relative_path is good.
    # In builder logic, we know the relative path from the root data dir ideally.
    # But here we just assume pdf_dir is the root.

    # Let's assume pdf_dir passed to build is the root for relative paths.
    # For now, let's look up by hash to find exact duplicate content, OR look up by filename?
    # unique=True on file_name in models.py suggests unique filenames.
    # But usually relative_path is safer.

    # Let's search by file_name for now as per models.py uniqueness?
    # models.py: file_name: Mapped[str] = Column(String, unique=True, nullable=False)
    # This might be restrictive if folder structure matters.
    # But let's stick to it.

    stmt = select(Manual).where(Manual.file_name == pdf_path.name)
    manual = session.execute(stmt).scalars().first()

    if manual:
        logger.info(f"Manual {pdf_path.name} found in DB. Checking hash...")
        if manual.file_hash == file_hash:
            logger.info("Hash match. Skipping DB sync (bookmarks).")
            return manual, False
        else:
            logger.info("Hash mismatch. Updating...")
            # Delete old bookmarks
            for bm in manual.bookmarks:
                session.delete(bm)
    else:
        logger.info(f"Creating new Manual entry for {pdf_path.name}")
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
    session = SessionLocal()

    # Initialize Chroma components
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

    # Initialize Docling converter
    logger.info("Initializing Docling converter...")
    converter = DocumentConverter()

    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")

        try:
            # 1. Sync to Relational DB
            manual, updated = sync_manual_to_db(session, pdf_path, pdf_dir)

            if not updated and not reset:
                logger.info(
                    f"File {pdf_path.name} unchanged. Skipping DB registration processing."
                )
                continue

            # 2. Convert to Markdown (Docling)
            # We need the doc object for chunking
            result = converter.convert(str(pdf_path))

            # Save Markdown (optional)
            if save_markdown:
                md_content = result.document.export_to_markdown()
                rel_path = pdf_path.relative_to(pdf_dir)
                md_path = settings.MARKDOWN_OUTPUT_DIR / rel_path.with_suffix(".md")
                md_path.parent.mkdir(parents=True, exist_ok=True)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)

            # 3. Coordinate-Based Chunking
            chunks = chunk_text_by_coordinates(result.document, manual)
            logger.info(f"Generated {len(chunks)} chunks for {pdf_path.name}")

            if not chunks:
                continue

            # 4. Add to ChromaDB
            ids = [f"{manual.id}_{i}" for i in range(len(chunks))]
            documents = [c["text"] for c in chunks]
            metadatas = []

            rel_path = pdf_path.relative_to(
                pdf_dir
            )  # Needed for metadata source if we want it relative

            for c in chunks:
                meta = {
                    "source": str(rel_path),
                    "manual_id": str(manual.id),
                }

                if c["metadata"].get("bookmark_id"):
                    meta["bookmark_id"] = str(c["metadata"]["bookmark_id"])

                metadatas.append(meta)

            collection.add(ids=ids, documents=documents, metadatas=metadatas)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            continue

    session.close()
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
