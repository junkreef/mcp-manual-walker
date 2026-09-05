import argparse
import contextlib
import io
import json
import logging
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import zstandard
from sqlalchemy import select

from mcp_manual_walker import lexical
from mcp_manual_walker.config import settings
from mcp_manual_walker.database import SessionLocal, init_db
from mcp_manual_walker.embeddings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_METADATA_KEY,
    check_collection_model,
    collection_metadata,
    get_embedder,
)
from mcp_manual_walker.models import Bookmark, Figure, Manual
from mcp_manual_walker.tui import watch

# chromadb is imported where it is used, for the same reason as the builder:
# `watch` needs neither.
chromadb = None


def _load_chromadb():
    global chromadb
    if chromadb is None:
        try:
            import chromadb as _chromadb
        except ImportError:
            return None
        chromadb = _chromadb
    return chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_manager")

# Archive layout version: 1 had no figures, 2 adds the "figures" records in
# sqlite.json plus one PNG per figure under figures/ in the zip, 3 replaces
# chroma.json with chunks.jsonl -- one chunk per line, so neither side has to
# hold the whole corpus. A 504,346-chunk export was killed by the OOM killer
# building the single JSON string: 3.8 GB of Python float lists plus roughly
# 5.8 GB of JSON text for the vectors alone. 4 compresses that line stream
# with zstd instead of deflate and stores the PNGs uncompressed, which is what
# lets both sides stream it straight in and out of the archive: no side now
# writes a multi-gigabyte temporary file. Versions 2 and 3 still import.
EXPORT_FORMAT_VERSION = 4
CHUNKS_FILE_NAME = "chunks.jsonl"
CHUNKS_ZST_FILE_NAME = "chunks.jsonl.zst"
CHROMA_FILE_NAME = "chroma.json"

# The chunk stream is the archive: 10.1 GB of the 12.3 GB in a zOS/V3R1
# export, almost all of it vector text. Both of these are the whole corpus,
# not a projection:
#
#     deflate-6 (version 3)   1,942 MB   one core, roughly 25 minutes
#     zstd-6    (version 4)   1,517 MB   every core, inside an 8:32 export
#
# threads=-1 uses every core; deflate could only ever use one, which is what
# made version 3 slow. On a representative 383 MB slice, 8 threads:
#
#     level 6     1.63 s   6.37x    <- this
#     level 9     3.03 s   6.62x    +37 s over the corpus to save 60 MB
#     level 12    7.57 s   6.46x    worse than 9 on both axes
#
# Long-distance matching (window_log 27) was measured too and gains nothing:
# embedding text has no long-range repeats to find.
#
# Beware of sampling this corpus to re-tune the level. Its first 20,000 chunks
# compress at 8.83x under zstd but 6.37x in the middle, so a head sample
# understates the finished archive by a third. Deflate barely varies over the
# same slices (5.33x / 5.10x / 5.13x), which is why checking such a sample
# against deflate does not catch it.
CHUNK_COMPRESS_LEVEL = 6

# The figure PNGs are the other 2.1 GB, and they are already compressed:
# deflating 2,140 MB of them saved 140 MB for roughly a third of the export's
# runtime. They go in stored -- which is why the archive only shrank from
# 3.96 GB to 3.68 GB even though the chunk stream lost 425 MB.
FIGURE_COMPRESS_TYPE = zipfile.ZIP_STORED

# Chunks handed to Chroma in one add() on import.
CHUNK_IMPORT_BATCH = 2000

# Chunks read back from Chroma in one page when rebuilding the lexical index.
LEXICAL_REINDEX_BATCH = 5000
FIGURES_DIR_NAME = "figures"


def get_chroma_client():
    module = _load_chromadb()
    if module is None:
        logger.error("chromadb is not installed.")
        sys.exit(1)
    return module.PersistentClient(path=str(settings.CHROMADB_PATH))


def command_build(args):
    """
    Build the database from PDFs.
    """
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    # Imported here, not at module scope: the builder pulls in Docling, torch,
    # transformers and PIL, which is 936 MB of resident memory. Every other
    # subcommand -- `watch` above all, which reads a text file and draws it --
    # was paying that. `watch` runs alongside a build that is already close to
    # the host's memory ceiling, so the cost lands exactly where there is none
    # to spare.
    from mcp_manual_walker.builder import build

    progress_file = None if args.no_progress else Path(args.progress_file)
    if args.include:
        logger.info(f"Starting build from {pdf_dir} (include={args.include})...")
    else:
        logger.info(f"Starting build from {pdf_dir}...")
    if progress_file:
        logger.info(
            f"Progress: uv run db_manager watch --progress-file {progress_file}"
        )
    build(
        pdf_dir,
        args.reset,
        args.save_markdown,
        args.include,
        progress_file,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
    )


def command_watch(args):
    """Watch a build's progress file."""
    sys.exit(
        watch(
            Path(args.progress_file),
            once=args.once,
            exit_when_finished=args.exit_when_finished,
        )
    )


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
            payload = [manual_to_dict(m, include_figures=False) for m in manuals]
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(f"{'ID':<38} | {'Pages':<6} | {'Figs':<5} | {'Path'}")
            print("-" * 80)
            for manual in manuals:
                print(
                    f"{manual.id:<38} | {manual.page_count:<6} | "
                    f"{len(manual.figures):<5} | {manual.relative_path}"
                )

    finally:
        session.close()


def manual_to_dict(manual, include_figures: bool = True):
    """Serializes a manual; ``include_figures`` adds the figure records."""
    data = {
        "id": manual.id,
        "file_name": manual.file_name,
        "document_title": manual.document_title,
        "relative_path": manual.relative_path,
        "file_hash": manual.file_hash,
        "page_count": manual.page_count,
        "updated_at": manual.updated_at.isoformat() if manual.updated_at else None,
        "figure_count": len(manual.figures),
        "bookmarks": [bookmark_to_dict(bm) for bm in manual.bookmarks],
    }
    if include_figures:
        data["figures"] = [figure_to_dict(f) for f in manual.figures]
    return data


def figure_to_dict(figure):
    """Serializes a figure; the PNG bytes travel as a separate zip member."""
    return {
        "id": figure.id,
        "manual_id": figure.manual_id,
        "bookmark_id": figure.bookmark_id,
        "picture_index": figure.picture_index,
        "page": figure.page,
        "bbox_l": figure.bbox_l,
        "bbox_b": figure.bbox_b,
        "bbox_r": figure.bbox_r,
        "bbox_t": figure.bbox_t,
        "caption": figure.caption,
        "labels": figure.labels,
        "description": figure.description,
        "mime_type": figure.mime_type,
        "width": figure.width,
        "height": figure.height,
        "created_at": figure.created_at.isoformat() if figure.created_at else None,
        "image_file": f"figures/{figure.id}.png",
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


def _write_chunks(collection, manual_ids: list[str], out) -> int:
    """Writes every chunk of the given manuals to `out`, one JSON line each.

    `out` is any text stream; the export points it at a zstd compressor
    feeding a zip member, so the corpus never lands on disk uncompressed.

    One manual is read at a time and written out before the next is read, so
    the peak is one manual's chunks rather than the corpus.

    Reading one manual per query is also the only size that always fits: a
    `$in` over several fails once the *matching chunks* exceed SQLite's
    parameter limit, not the ids -- measured on this corpus, 10 ids matching
    11,125 chunks was fine and 20 ids was not, and a manual holds anywhere
    from 328 to several thousand chunks.

    A chunk with no embedding is written anyway, with a null: dropping it here
    would silently change what the archive contains, and the importer already
    knows what to do about it.
    """
    written = 0
    for index, manual_id in enumerate(manual_ids, start=1):
        got = collection.get(
            where={"manual_id": manual_id},
            include=["embeddings", "metadatas", "documents"],
        )
        embeddings = got.get("embeddings")
        for i, chunk_id in enumerate(got["ids"]):
            emb = None
            if embeddings is not None and i < len(embeddings):
                emb = embeddings[i]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
            out.write(
                json.dumps(
                    {
                        "id": chunk_id,
                        "embedding": emb,
                        "metadata": got["metadatas"][i],
                        "document": got["documents"][i],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
        if index % 50 == 0 or index == len(manual_ids):
            logger.info(
                f"Wrote {written:,} chunk(s) from {index}/{len(manual_ids)} manuals."
            )
    return written


def _read_chunks(lines):
    """Yields one chunk per non-blank line of `lines`."""
    for line in lines:
        line = line.strip()
        if line:
            yield json.loads(line)


@contextlib.contextmanager
def _chunk_source(zf: zipfile.ZipFile):
    """Yields an iterator over an archive's chunks, whatever version wrote it.

    Every branch reads straight out of the open archive. Version 3 unpacked
    its 10 GB chunks.jsonl to a temporary file first, which meant an import
    needed room for the archive *and* its contents; nothing here does.
    """
    names = set(zf.namelist())
    if CHUNKS_ZST_FILE_NAME in names:
        with zf.open(CHUNKS_ZST_FILE_NAME) as member:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(member) as raw:
                yield _read_chunks(io.TextIOWrapper(raw, encoding="utf-8"))
    elif CHUNKS_FILE_NAME in names:
        with zf.open(CHUNKS_FILE_NAME) as member:
            yield _read_chunks(io.TextIOWrapper(member, encoding="utf-8"))
    elif CHROMA_FILE_NAME in names:
        yield _chunks_from_legacy(json.loads(zf.read(CHROMA_FILE_NAME)))
    else:
        logger.error(
            f"Archive contains none of {CHUNKS_ZST_FILE_NAME}, "
            f"{CHUNKS_FILE_NAME} or {CHROMA_FILE_NAME}."
        )
        sys.exit(1)


def _chunks_from_legacy(chroma_data: dict):
    """Yields the chunks of a version-2 archive in the same shape."""
    embeddings = chroma_data.get("embeddings") or []
    for i, chunk_id in enumerate(chroma_data.get("ids") or []):
        yield {
            "id": chunk_id,
            "embedding": embeddings[i] if i < len(embeddings) else None,
            "metadata": chroma_data["metadatas"][i],
            "document": chroma_data["documents"][i],
        }


def _stored_info(name: str) -> zipfile.ZipInfo:
    """A zip member written verbatim, for data that is already compressed."""
    info = zipfile.ZipInfo(name, date_time=datetime.now().timetuple()[:6])
    info.compress_type = FIGURE_COMPRESS_TYPE
    return info


def _write_chunk_member(zf: zipfile.ZipFile, collection, manual_ids) -> int:
    """Streams every chunk into the archive as one zstd-compressed member.

    The member itself is stored, because its bytes are already a zstd frame.
    Chroma is read, serialized, compressed and written in one pass, so the
    corpus is never held in memory nor spilled to a temporary file.
    """
    info = _stored_info(CHUNKS_ZST_FILE_NAME)
    compressor = zstandard.ZstdCompressor(
        level=CHUNK_COMPRESS_LEVEL, threads=-1
    )
    with zf.open(info, "w", force_zip64=True) as member:
        with compressor.stream_writer(member, closefd=False) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
            try:
                count = _write_chunks(collection, manual_ids, text)
                text.flush()
            finally:
                # Detach rather than close: closing the wrapper would close
                # the compressor under it and end the frame twice.
                text.detach()
    return count


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

        figure_count = 0

        # Every member is written straight into the archive as it is produced.
        # Nothing here is staged in a temporary directory: for this corpus that
        # staging was 12.3 GB of writes for a 3.4 GB result.
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            chunk_count = _write_chunk_member(zf, collection, manual_ids)

            for manual in manuals:
                for figure in manual.figures:
                    zf.writestr(
                        _stored_info(f"{FIGURES_DIR_NAME}/{figure.id}.png"),
                        figure.image,
                    )
                    figure_count += 1

            zf.writestr(
                "sqlite.json", json.dumps(sqlite_data, indent=2)
            )
            # Last, because the chunk count is only known once they are
            # written. A reader finds it through the central directory, which
            # does not care what order the members came in.
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "version": "1.0",
                        "format_version": EXPORT_FORMAT_VERSION,
                        "created_at": datetime.now().isoformat(),
                        "target": target,
                        "manual_count": len(manuals),
                        "chunk_count": chunk_count,
                        "figure_count": figure_count,
                        # Vectors are only reusable by the model that made them.
                        EMBEDDING_MODEL_METADATA_KEY: settings.EMBEDDING_MODEL,
                    },
                    indent=2,
                ),
            )

        logger.info(
            f"Export completed: {output_path} "
            f"({chunk_count:,} chunk(s), {figure_count} figure image(s))"
        )

    finally:
        session.close()


def _read_figure_images(zf: zipfile.ZipFile, sqlite_data) -> dict:
    """Reads the exported figure PNGs straight out of the archive, by id."""
    names = set(zf.namelist())
    images = {}
    for manual_dict in sqlite_data:
        for fig_dict in manual_dict.get("figures", []):
            member = fig_dict.get("image_file")
            if not member or member not in names:
                logger.warning(
                    f"Figure {fig_dict.get('id')} has no image member "
                    f"'{member}' in the archive. Skipping it."
                )
                continue
            images[fig_dict["id"]] = zf.read(member)
    return images


def command_import(args):
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Importing from {input_path}")

    session = SessionLocal()
    client = get_chroma_client()

    try:
        # The archive stays open for the whole import and every member is read
        # out of it in place. Unpacking it first meant needing room for the
        # archive and its contents at once -- 12.3 GB of /tmp for zOS/V3R1.
        with zipfile.ZipFile(input_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))

            # Archives written before figures existed carry no format_version.
            format_version = manifest.get("format_version", 1)
            logger.info(
                f"Importing export from {manifest['created_at']}, target: {manifest['target']}"
            )
            logger.info(f"Archive format version: {format_version}")

            sqlite_data = json.loads(zf.read("sqlite.json"))
            figure_images = _read_figure_images(zf, sqlite_data)

            # Imported vectors are only usable with the model that produced them.
            exported_model = manifest.get(EMBEDDING_MODEL_METADATA_KEY)
            if exported_model is None:
                logger.warning(
                    "The export does not record an embedding model. Its "
                    "vectors may have been built with a model other than "
                    f"{settings.EMBEDDING_MODEL}."
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
            imported_figures = 0
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

                for fig_dict in manual_dict.get("figures", []):
                    image = figure_images.get(fig_dict["id"])
                    if image is None:
                        # Already reported by _read_figure_images.
                        continue
                    session.add(
                        Figure(
                            id=fig_dict["id"],
                            manual_id=fig_dict["manual_id"],
                            bookmark_id=fig_dict.get("bookmark_id"),
                            picture_index=fig_dict["picture_index"],
                            page=fig_dict["page"],
                            bbox_l=fig_dict["bbox_l"],
                            bbox_b=fig_dict["bbox_b"],
                            bbox_r=fig_dict["bbox_r"],
                            bbox_t=fig_dict["bbox_t"],
                            caption=fig_dict.get("caption"),
                            labels=fig_dict.get("labels"),
                            description=fig_dict.get("description"),
                            mime_type=fig_dict.get("mime_type", "image/png"),
                            width=fig_dict.get("width"),
                            height=fig_dict.get("height"),
                            created_at=datetime.fromisoformat(fig_dict["created_at"])
                            if fig_dict.get("created_at")
                            else None,
                            image=image,
                        )
                    )
                    imported_figures += 1

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
                f"SQLite Import: {imported_count} imported, {skipped_count} skipped, "
                f"{imported_figures} figure(s) restored."
            )

            # Import Chroma Data, a batch at a time. An archive of this corpus
            # holds half a million chunks; accumulating them before the first
            # add() is the same mistake that made the export side run out of
            # memory.
            added, skipped_chunks = 0, 0
            pending: dict[str, list] = {
                "ids": [],
                "embeddings": [],
                "metadatas": [],
                "documents": [],
            }

            # The lexical index is derived data and is never carried in an
            # archive: rebuilding it here costs seconds against the tens of
            # minutes the import already takes, and it keeps the archive
            # format free of anything that is only a search implementation
            # detail.
            lexical_conn = lexical.sqlite_connection(session)
            lexical.create_table(lexical_conn)

            def flush_chunks():
                nonlocal added, pending
                if not pending["ids"]:
                    return
                collection.add(
                    ids=pending["ids"],
                    embeddings=pending["embeddings"],
                    metadatas=pending["metadatas"],
                    documents=pending["documents"],
                )
                lexical.add_chunks(
                    lexical_conn,
                    (
                        (cid, meta.get("manual_id") or "", doc or "")
                        for cid, meta, doc in zip(
                            pending["ids"], pending["metadatas"], pending["documents"]
                        )
                    ),
                )
                added += len(pending["ids"])
                # Fresh lists rather than clear(): the ones just handed to add()
                # belong to the caller now.
                pending = {
                    "ids": [],
                    "embeddings": [],
                    "metadatas": [],
                    "documents": [],
                }

            with _chunk_source(zf) as source:
                for chunk in source:
                    meta = chunk["metadata"]
                    # Only chunks whose manual was imported: an archive can overlap
                    # a database that already holds part of it.
                    if meta.get("manual_id") not in accepted_manual_ids:
                        continue
                    if not chunk.get("embedding"):
                        logger.warning(
                            f"Skipping import of chunk {chunk['id']}: "
                            "missing embedding."
                        )
                        skipped_chunks += 1
                        continue
                    pending["ids"].append(chunk["id"])
                    pending["embeddings"].append(chunk["embedding"])
                    pending["metadatas"].append(meta)
                    pending["documents"].append(chunk["document"])
                    if len(pending["ids"]) >= CHUNK_IMPORT_BATCH:
                        flush_chunks()
                flush_chunks()

            if added:
                logger.info(f"ChromaDB Import: Added {added:,} chunks.")
            else:
                logger.info("ChromaDB Import: No chunks to add (all skipped or empty).")
            if skipped_chunks:
                logger.warning(f"{skipped_chunks:,} chunk(s) had no embedding.")

            if added:
                lexical.optimize(lexical_conn)
                session.commit()
                logger.info(f"Lexical index: {added:,} chunk(s) indexed.")

    finally:
        session.close()


def command_reindex_lexical(args):
    """Rebuilds the BM25 index from the chunks already in ChromaDB.

    The lexical index is derived data, so it is not in an export archive and a
    database made before it existed simply has none. Rebuilding it from Chroma
    takes a couple of minutes against the twenty an import costs, so this is
    the way to add it to a database that is otherwise fine.
    """
    session = SessionLocal()
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        conn = lexical.sqlite_connection(session)
        lexical.drop_table(conn)
        lexical.create_table(conn)

        total = collection.count()
        logger.info(f"Rebuilding the lexical index over {total:,} chunk(s)...")
        written = offset = 0
        while True:
            got = collection.get(
                limit=LEXICAL_REINDEX_BATCH,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if not got["ids"]:
                break
            written += lexical.add_chunks(
                conn,
                (
                    (cid, (meta or {}).get("manual_id") or "", doc or "")
                    for cid, meta, doc in zip(
                        got["ids"], got["metadatas"], got["documents"]
                    )
                ),
            )
            offset += len(got["ids"])
            if offset % (LEXICAL_REINDEX_BATCH * 10) == 0:
                logger.info(f"  {offset:,}/{total:,}")
        lexical.optimize(conn)
        session.commit()
        logger.info(f"Lexical index rebuilt: {written:,} chunk(s).")
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

            # Delete from SQLite (cascades to bookmarks and figures)
            session.delete(manual)
            logger.info("  - Deleted from SQLite (bookmarks and figures included)")

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
    parser_build.add_argument(
        "--include",
        action="append",
        metavar="GLOB",
        help=(
            "Only build PDFs whose path relative to --pdf_dir matches this glob. "
            "Repeatable; a file matching any pattern is kept. These are fnmatch "
            "patterns, so '*' also matches '/': --include 'zOS/V3R1/*' takes that "
            "directory and everything below it. Keep --pdf_dir at the corpus root "
            "so stored paths stay consistent across subset builds."
        ),
    )
    parser_build.add_argument(
        "--progress-file",
        type=str,
        default=str(settings.BUILD_PROGRESS_FILE),
        help=(
            "Where to write the per-file progress log that `db_manager watch` "
            "reads (default: BUILD_PROGRESS_FILE). Truncated at the start of "
            "every build."
        ),
    )
    parser_build.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not write a progress log.",
    )
    parser_build.add_argument(
        "--min-pages",
        type=int,
        metavar="N",
        help=(
            "Only convert documents with at least N pages. With --max-pages, "
            "splits a corpus by document length so the biggest manuals can be "
            "run at a lower DOCLING_WORKERS than the rest: peak memory per "
            "worker is set by the length of the document it holds, so one pass "
            "sized for the longest manual wastes the machine on all the others."
        ),
    )
    parser_build.add_argument(
        "--max-pages",
        type=int,
        metavar="N",
        help="Only convert documents with at most N pages.",
    )
    parser_build.set_defaults(func=command_build)

    # Watch Command
    parser_watch = subparsers.add_parser(
        "watch", help="Live view of a build's per-file progress"
    )
    parser_watch.add_argument(
        "--progress-file",
        type=str,
        default=str(settings.BUILD_PROGRESS_FILE),
        help="Progress log to read (default: BUILD_PROGRESS_FILE).",
    )
    parser_watch.add_argument(
        "--once",
        action="store_true",
        help="Print a single snapshot and exit instead of watching.",
    )
    parser_watch.add_argument(
        "--exit-when-finished",
        action="store_true",
        help="Quit on the run's final event rather than waiting for 'q'.",
    )
    parser_watch.set_defaults(func=command_watch)

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

    # Lexical reindex Command
    parser_lex = subparsers.add_parser(
        "reindex-lexical",
        help="Rebuild the BM25 index from the chunks already in ChromaDB",
    )
    parser_lex.set_defaults(func=command_reindex_lexical)

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
