import argparse
import logging
import shutil
import sys
import warnings
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

# Suppress warnings from libraries
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("builder")

# Imports for dependencies
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
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

# Imports for DB Sync
from sqlalchemy.orm import Session
from sqlalchemy import select

# Local imports
try:
    from mcp_manual_walker.config import settings
    from mcp_manual_walker.database import init_db, SessionLocal, engine
    from mcp_manual_walker.models import Manual, Bookmark, Base
    from mcp_manual_walker.pdf_utils import calculate_file_hash, extract_pdf_metadata
    from mcp_manual_walker.chunking import chunk_text_by_coordinates
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
    # Helper to get the embedding function
    # Using intfloat/multilingual-e5-small as requested
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )

def sync_manual_to_db(session: Session, pdf_path: Path, output_dir: Path) -> Manual:
    """
    Syncs the Manual and Bookmarks to the SQLite DB.
    """
    rel_path = pdf_path.relative_to(pdf_path.parent.parent.parent) # Hacky?
    # Better: user provided pdf_dir. rel_path = pdf_path.relative_to(pdf_dir)
    # But pdf_dir is passed to build(). We should pass it here or calculate hash.
    
    # We will use pdf_utils' approach or just use the hash.
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
            return manual
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
    # manual.relative_path ... we need a consistent relative path.
    # For now, store absolute or name? models.py says relative_path is required.
    # We'll set it to filename if not strictly defined, or use the path relative to pdf_dir (need to pass pdf_dir).
    # Since we don't have pdf_dir in args here easily without changing signature, let's just use parent name + filename
    manual.relative_path = f"{pdf_path.parent.name}/{pdf_path.name}"
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
            page_top=bm_data.get("top"), # From our enhanced pdf_utils
            parent_id=parent_id
        )
        session.add(new_bm)
        
        # Push to stack
        parent_stack.append((level, new_bm))
        
    session.commit()
    logger.info(f"Synced {len(bookmarks_data)} bookmarks to DB.")
    return manual


def build(pdf_dir: Path, output_dir: Path, reset: bool):
    check_dependencies()
    
    # Prepare output directories
    if reset:
        if output_dir.exists():
            logger.warning(f"Resetting output directory: {output_dir}")
            shutil.rmtree(output_dir)
        # Also reset DB if it's the default one in data/
        # But DB might be shared.
        # Ideally, we drop tables or delete rows for these manuals.
        # Since we use --reset mostly for full rebuild, let's allow it to clear Chroma but maybe strictly manage SQL?
        # Creating tables is handled by init_db.
    
        if settings.DB_FILE_PATH.exists():
             logger.warning(f"Deleting existing DB: {settings.DB_FILE_PATH}")
             settings.DB_FILE_PATH.unlink()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "markdown").mkdir(exist_ok=True)
    
    # Initialize DB
    logger.info(f"Initializing Database at {settings.DB_FILE_PATH}...")
    init_db()
    session = SessionLocal()
    
    # Initialize Chroma components
    logger.info("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(output_dir / "chroma_db"))
    embedding_fn = get_embedding_function()
    
    collection = client.get_or_create_collection(
        name="manual_chunks",
        embedding_function=embedding_fn,
        metadata={"description": "Chunks from PDF manuals"}
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
            manual = sync_manual_to_db(session, pdf_path, output_dir)
            
            # 2. Convert to Markdown (Docling)
            # We need the doc object for chunking
            result = converter.convert(str(pdf_path))
            
            # Save Markdown (optional/backup)
            md_content = result.document.export_to_markdown()
            rel_path = pdf_path.relative_to(pdf_dir)
            md_path = output_dir / "markdown" / rel_path.with_suffix(".md")
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
            
            base_metadata = {
                "source": str(rel_path),
                "manual_id": str(manual.id),
                "page": 0 # Default/Placeholder, real page in chunk? 
                # Chunk metadata has manual_id and bookmark_id.
                # Do we need page number per chunk? 
                # Yes, but chunks might span pages? 
                # Our chunking logic splits by bookmark, but text flows.
                # Let's check simulating_chunking output. It included page.
                # Update chunking.py to clear this up? 
                # chunking.py returns list of dicts.
            }
            
            for c in chunks:
                # meta = base_metadata.copy() # Base + specific
                # c["metadata"] has manual_id and bookmark_id
                # Let's ensure we have valid metadata types for Chroma (str, int, float, bool)
                # bookmark_id can be None. Chroma doesn't like None values sometimes?
                # Chroma requires metadata values to be str, int, float, bool. None removes the key.
                
                meta = {
                    "source": str(rel_path),
                    "manual_id": str(manual.id),
                }
                
                if c["metadata"].get("bookmark_id"):
                    meta["bookmark_id"] = str(c["metadata"]["bookmark_id"])
                
                # Add page info if available?
                # chunking.py returns text but lost page info in aggregation?
                # current_text_buffer aggregates text.
                # If we want page number, we should aggregate that too, or take start page.
                
                metadatas.append(meta)
                
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            continue
        
    session.close()
    logger.info("Build complete.")
    logger.info(f"Database saved to {output_dir / 'chroma_db'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB from PDFs using Docling.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output (DB and Markdown).")
    parser.add_argument("--reset", action="store_true", help="Delete output directory before starting.")
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    build(pdf_dir, output_dir, args.reset)
