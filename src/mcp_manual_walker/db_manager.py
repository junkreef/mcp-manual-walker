import argparse
import json
import logging
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from mcp_manual_walker.builder import build
from mcp_manual_walker.config import settings
from mcp_manual_walker.database import SessionLocal, init_db
from mcp_manual_walker.embeddings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_METADATA_KEY,
    check_collection_model,
    collection_metadata,
    get_embedder,
)
from mcp_manual_walker.models import Bookmark, Manual

try:
    import chromadb
except ImportError:
    chromadb = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_manager")


def get_chroma_client():
    if chromadb is None:
        logger.error("chromadb is not installed.")
        sys.exit(1)
    return chromadb.PersistentClient(path=str(settings.CHROMADB_PATH))


def command_build(args):
    """
    Build the database from PDFs.
    """
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    logger.info(f"Starting build from {pdf_dir}...")
    build(pdf_dir, args.reset, args.save_markdown)


def command_list(args):
    """
    List all registered manuals.
    """
    session = SessionLocal()
    try:
        stmt = select(Manual).order_by(Manual.relative_path)
        manuals = session.execute(stmt).scalars().all()

        if not manuals:
            print("No manuals found in the database.")
            return

        if args.json:
            print(
                json.dumps([manual_to_dict(m) for m in manuals], indent=2, default=str)
            )
        else:
            print(f"{'ID':<38} | {'Pages':<6} | {'Path'}")
            print("-" * 80)
            for manual in manuals:
                print(
                    f"{manual.id:<38} | {manual.page_count:<6} | {manual.relative_path}"
                )

    finally:
        session.close()


def manual_to_dict(manual):
    return {
        "id": manual.id,
        "file_name": manual.file_name,
        "document_title": manual.document_title,
        "relative_path": manual.relative_path,
        "file_hash": manual.file_hash,
        "page_count": manual.page_count,
        "updated_at": manual.updated_at.isoformat() if manual.updated_at else None,
        "bookmarks": [bookmark_to_dict(bm) for bm in manual.bookmarks],
    }


def bookmark_to_dict(bookmark):
    return {
        "id": bookmark.id,
        "manual_id": bookmark.manual_id,
        "ordering": bookmark.ordering,
        "title": bookmark.title,
        "level": bookmark.level,
        "page_num": bookmark.page_num,
        "page_top": bookmark.page_top,
        "parent_id": bookmark.parent_id,
    }


def command_export(args):
    target = args.target
    output_path = Path(args.output)

    logger.info(f"Exporting manuals matching target: {target}")

    session = SessionLocal()
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    try:
        stmt = select(Manual).where(Manual.relative_path.startswith(target))
        manuals = session.execute(stmt).scalars().all()

        if not manuals:
            logger.warning(f"No manuals found matching target: {target}")
            return

        manual_ids = [m.id for m in manuals]
        logger.info(f"Found {len(manuals)} manuals to export.")

        # Prepare SQLite data
        sqlite_data = [manual_to_dict(m) for m in manuals]

        # Prepare Chrome data
        # Fetching in batches might be better if huge, but let's assume it fits in memory for now.
        chroma_results = collection.get(
            where={"manual_id": {"$in": manual_ids}},
            include=["embeddings", "metadatas", "documents"],
        )

        # Convert numpy arrays to lists
        embeddings = chroma_results.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            # Check if it's a valid list (it could be None)
            # If items are numpy arrays, convert them.
            # chroma_results["embeddings"] is usually a list of lists or list of numpy arrays.
            # safe conversion:
            embeddings_list = []
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(emb)
        else:
            embeddings_list = None

        chroma_data = {
            "ids": chroma_results["ids"],
            "embeddings": embeddings_list,
            "metadatas": chroma_results["metadatas"],
            "documents": chroma_results["documents"],
        }

        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "target": target,
            "manual_count": len(manuals),
            "chunk_count": len(chroma_results["ids"]),
            # Vectors are only reusable by the model that produced them.
            EMBEDDING_MODEL_METADATA_KEY: settings.EMBEDDING_MODEL,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with open(temp_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            with open(temp_path / "sqlite.json", "w", encoding="utf-8") as f:
                json.dump(sqlite_data, f, indent=2)

            with open(temp_path / "chroma.json", "w", encoding="utf-8") as f:
                json.dump(chroma_data, f)

            # Create Zip
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_name in ["manifest.json", "sqlite.json", "chroma.json"]:
                    zf.write(temp_path / file_name, arcname=file_name)

        logger.info(f"Export completed: {output_path}")

    finally:
        session.close()


def command_import(args):
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Importing from {input_path}")

    session = SessionLocal()
    client = get_chroma_client()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(temp_path)

            with open(temp_path / "manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)

            logger.info(
                f"Importing export from {manifest['created_at']}, target: {manifest['target']}"
            )

            with open(temp_path / "sqlite.json", "r", encoding="utf-8") as f:
                sqlite_data = json.load(f)

            with open(temp_path / "chroma.json", "r", encoding="utf-8") as f:
                chroma_data = json.load(f)

        # Imported vectors are only usable with the model that produced them.
        exported_model = manifest.get(EMBEDDING_MODEL_METADATA_KEY)
        if exported_model is None:
            logger.warning(
                "The export does not record an embedding model. Its vectors may "
                f"have been built with a model other than {settings.EMBEDDING_MODEL}."
            )
        elif exported_model != settings.EMBEDDING_MODEL:
            logger.error(
                f"The export was built with '{exported_model}', but "
                f"settings.EMBEDDING_MODEL is '{settings.EMBEDDING_MODEL}'. "
                "Import aborted: re-export the data with the current model."
            )
            return

        # Load the embedding model only to stamp the collection metadata; every
        # vector comes from the archive, so Chroma never embeds anything here.
        logger.info("Loading embedding model...")
        embedder = get_embedder()

        if embedder:
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                metadata=collection_metadata(embedder),
            )
        else:
            # Without the model we cannot stamp the model name on a freshly
            # created collection; an existing one keeps its own metadata.
            logger.warning(
                "Embedding model unavailable: a newly created collection will "
                "not record which model built its vectors."
            )
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
            )

        # Import SQLite Data
        imported_count = 0
        skipped_count = 0
        accepted_manual_ids = set()

        for manual_dict in sqlite_data:
            # Check existence
            existing = session.scalars(select(Manual).where(Manual.id == manual_dict["id"])).first()
            if existing:
                logger.warning(
                    f"Manual {manual_dict['relative_path']} (ID: {manual_dict['id']}) already exists. Skipping."
                )
                skipped_count += 1
                continue

            manual = Manual(
                id=manual_dict["id"],
                file_name=manual_dict["file_name"],
                document_title=manual_dict["document_title"],
                relative_path=manual_dict["relative_path"],
                file_hash=manual_dict["file_hash"],
                page_count=manual_dict["page_count"],
                updated_at=datetime.fromisoformat(manual_dict["updated_at"])
                if manual_dict["updated_at"]
                else None,
            )
            session.add(manual)
            accepted_manual_ids.add(manual.id)

            for bm_dict in manual_dict["bookmarks"]:
                bm = Bookmark(
                    id=bm_dict["id"],
                    manual_id=bm_dict["manual_id"],
                    ordering=bm_dict["ordering"],
                    title=bm_dict["title"],
                    level=bm_dict["level"],
                    page_num=bm_dict["page_num"],
                    page_top=bm_dict["page_top"],
                    parent_id=bm_dict["parent_id"],
                )
                session.add(bm)

            imported_count += 1

        session.commit()
        logger.info(
            f"SQLite Import: {imported_count} imported, {skipped_count} skipped."
        )

        # Import Chroma Data
        if chroma_data["ids"]:
            ids_to_add = []
            embeddings_to_add = []
            metadatas_to_add = []
            documents_to_add = []

            for i, meta in enumerate(chroma_data["metadatas"]):
                mid = meta.get("manual_id")
                # Add only if we imported the manual (avoid duplication scenarios or partial updates)
                if mid in accepted_manual_ids:
                    # Check embedding availability
                    emb = None
                    if chroma_data["embeddings"] and i < len(chroma_data["embeddings"]):
                        emb = chroma_data["embeddings"][i]

                    if emb:
                        ids_to_add.append(chroma_data["ids"][i])
                        embeddings_to_add.append(emb)
                        metadatas_to_add.append(meta)
                        documents_to_add.append(chroma_data["documents"][i])
                    else:
                        logger.warning(
                            f"Skipping import of chunk {chroma_data['ids'][i]}: missing embedding."
                        )

            if ids_to_add:
                collection.add(
                    ids=ids_to_add,
                    embeddings=embeddings_to_add,
                    metadatas=metadatas_to_add,
                    documents=documents_to_add,
                )
                logger.info(f"ChromaDB Import: Added {len(ids_to_add)} chunks.")
            else:
                logger.info("ChromaDB Import: No chunks to add (all skipped or empty).")
        else:
            logger.info("ChromaDB Import: No data in export.")

    finally:
        session.close()


def command_delete(args):
    target = args.target
    logger.info(f"Attempting to delete targets matching: {target}")

    session = SessionLocal()
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    try:
        # Find manuals starting with the target string (directory or specific file)
        # Note: relative_path stored in DB is relative to the PDF root used during build.
        stmt = select(Manual).where(Manual.relative_path.startswith(target))
        manuals = session.execute(stmt).scalars().all()

        if not manuals:
            logger.warning(f"No manuals found matching target: {target}")
            return

        for manual in manuals:
            logger.info(f"Deleting manual: {manual.relative_path} (ID: {manual.id})")

            # Delete from ChromaDB
            try:
                collection.delete(where={"manual_id": manual.id})
                logger.info("  - Deleted from ChromaDB")
            except Exception as e:
                logger.error(f"  - Failed to delete from ChromaDB: {e}")

            # Delete from SQLite (cascades to bookmarks)
            session.delete(manual)
            logger.info("  - Deleted from SQLite")

        session.commit()
        logger.info("Deletion complete.")

    except Exception as e:
        logger.error(f"An error occurred during deletion: {e}")
        session.rollback()
    finally:
        session.close()


def command_search(args):
    """
    Search functionality for verification purposes.
    """
    query = args.query
    n_results = args.n_results

    logger.info(f"Searching for: '{query}'")

    client = get_chroma_client()

    logger.info("Loading embedding model...")
    embedder = get_embedder()
    if not embedder:
        logger.error("Could not load the embedding model.")
        return

    collection = client.get_collection(name=COLLECTION_NAME)

    try:
        check_collection_model(collection, settings.EMBEDDING_MODEL)
    except RuntimeError as e:
        logger.error(str(e))
        return

    # The query is embedded here and passed explicitly: the collection has no
    # embedding function of its own.
    query_embeddings = [embedder.embed_query(query)]

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nSearch Results for '{query}':")
    print("-" * 80)

    if not results["documents"] or not results["documents"][0]:
        print("No results found.")
        return

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]

        print(f"Rank {i + 1} (Distance: {dist:.4f})")
        print(f"Source: {meta.get('source')} (Manual ID: {meta.get('manual_id')})")
        if meta.get("bookmark_id"):
            print(f"Bookmark ID: {meta.get('bookmark_id')}")
        print(f"Content: {doc[:200]}..." if len(doc) > 200 else f"Content: {doc}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Manage MCP Manual Walker Database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build Command
    parser_build = subparsers.add_parser("build", help="Build DB from PDFs")
    parser_build.add_argument(
        "--pdf_dir", type=str, required=True, help="Directory containing PDF files"
    )
    parser_build.add_argument(
        "--save-markdown", action="store_true", help="Save intermediate Markdown files"
    )
    parser_build.add_argument(
        "--reset", action="store_true", help="Reset database before building"
    )
    parser_build.set_defaults(func=command_build)

    # List Command
    parser_list = subparsers.add_parser("list", help="List registered manuals")
    parser_list.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    parser_list.set_defaults(func=command_list)

    # Export Command
    parser_export = subparsers.add_parser("export", help="Export data")
    parser_export.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target file or directory path relative to PDF root",
    )
    parser_export.add_argument(
        "--output", type=str, required=True, help="Output ZIP file path"
    )
    parser_export.set_defaults(func=command_export)

    # Import Command
    parser_import = subparsers.add_parser("import", help="Import data")
    parser_import.add_argument(
        "--input", type=str, required=True, help="Input ZIP file path"
    )
    parser_import.set_defaults(func=command_import)

    # Delete Command
    parser_delete = subparsers.add_parser("delete", help="Delete data")
    parser_delete.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target file or directory path relative to PDF root",
    )
    parser_delete.set_defaults(func=command_delete)

    # Search Command
    parser_search = subparsers.add_parser("search", help="Search the database")
    parser_search.add_argument("query", type=str, help="Search query")
    parser_search.add_argument(
        "--n_results", type=int, default=5, help="Number of results to return"
    )
    parser_search.set_defaults(func=command_search)

    args = parser.parse_args()

    # Initialize DB (creates tables if needed, binds engine)
    if not settings.DB_FILE_PATH.parent.exists():
        settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()

    args.func(args)


if __name__ == "__main__":
    main()
