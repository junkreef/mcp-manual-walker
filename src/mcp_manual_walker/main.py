import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional

import chromadb
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import lexical
from .config import settings
from .database import SessionLocal, init_db
from .embeddings import COLLECTION_NAME, check_collection_model, get_embedder
from .models import Bookmark, Figure, Manual
from .schemas import (
    BookmarkNode,
    FigureInfo,
    FigureRef,
    ManualInfo,
    ManualMetadata,
    MarkdownContent,
    SearchResult,
    SearchResultItem,
)

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global ChromaDB client and collection
# Global AppState
class AppState:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        # Reason why the vector store could not be used, surfaced by the tools.
        self.init_error = None


app_state = AppState()


def init_vector_store() -> None:
    """
    Connects app_state to the persisted vector collection.

    Kept out of the lifespan so it can be re-run (the test suite reuses a single
    server object). A failure is recorded instead of raised, so the server still
    starts and every tool can explain why the vector store is unusable.
    """
    logger.info("Initializing ChromaDB...")
    app_state.chroma_client = None
    app_state.collection = None
    app_state.embedder = None
    app_state.init_error = None

    try:
        chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMADB_PATH.resolve())
        )

        # Load the embedding model (the same one the builder used)
        embedder = get_embedder()

        # Get the collection without an embedding function: queries are embedded
        # here and passed to Chroma explicitly.
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # Vectors built with another model are not comparable to ours.
        check_collection_model(collection, settings.EMBEDDING_MODEL)

        app_state.chroma_client = chroma_client
        app_state.embedder = embedder
        app_state.collection = collection

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        app_state.init_error = str(e)


def _require_collection():
    """Returns the collection, or raises a ToolError explaining why it is missing."""
    if app_state.collection is None:
        message = "Vector database is not initialized."
        if app_state.init_error:
            message = f"{message} {app_state.init_error}"
        raise ToolError(message)
    return app_state.collection


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server startup event handler."""
    logger.info("Initializing application...")
    # Ensure all necessary directories exist before initializing the database
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMADB_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    init_db()

    init_vector_store()

    yield


app = FastMCP(lifespan=lifespan)


@app.tool(
    name="list_manuals",
    description="""Provides a comprehensive list of all available manuals. This tool is
    the primary entry point for discovering content. It returns a list of all manuals
    found in the system, each with a unique ID that is required by other tools
    like `get_manual_metadata`.

    Workflow Example:
    1. Call `list_manuals()` to get a list of all available manuals.
    2. Identify the manual you are interested in from the list.
    3. Use the `id` of that manual to call `get_manual_metadata()` to retrieve its
       table of contents and other details.""",
    tags={"manual", "discovery"},
    annotations={"readOnlyHint": True},
)
def list_manuals() -> List[ManualInfo]:
    """Returns a list of all available manuals."""
    db: Session = SessionLocal()
    try:
        manuals = db.query(Manual).order_by(Manual.file_name).all()
        return [
            ManualInfo(
                id=m.id,
                file_name=m.file_name,
                document_title=m.document_title,
            )
            for m in manuals
        ]
    except Exception as e:
        logger.error(f"Error fetching list of manuals: {e}")
        raise ToolError(e)
    finally:
        db.close()


def _build_toc(bookmarks: list[Bookmark]) -> list[BookmarkNode]:
    """Builds a nested table of contents from a flat list of bookmarks."""
    toc = []
    bookmark_map = {
        bm.id: BookmarkNode(id=bm.id, title=bm.title, page=bm.page_num, children=[])
        for bm in bookmarks
    }
    for bm in bookmarks:
        if bm.parent_id:
            if parent := bookmark_map.get(bm.parent_id):
                parent.children.append(bookmark_map[bm.id])
        else:
            toc.append(bookmark_map[bm.id])
    return toc


@app.tool(
    name="get_manual_metadata",
    description="""Retrieves detailed metadata and a hierarchical table of contents for
    a specific manual. Use this tool after you have identified a manual of interest
    using `list_manuals()`. It provides the full structure of the manual's bookmarks,
    which is essential for navigating its content. Each bookmark in the table of
    contents has its own unique ID, which is required by the `get_markdown_content`
    tool to fetch the actual content of that section.

    Workflow Example:
    1. Get a `manual_id` from the output of `list_manuals()`.
    2. Call `get_manual_metadata(manual_id=...)` to get the manual's structure.
    3. Browse the `table_of_contents` to find the specific section you need.
    4. Use the `id` of the desired bookmark to call `get_markdown_content()`.""",
    tags={"manual", "metadata", "toc"},
    annotations={"readOnlyHint": True},
)
def get_manual_metadata(
    manual_id: Annotated[
        str,
        Field(
            description="""The unique ID of the manual, 
            obtained from the `list_manuals` tool."""
        ),
    ],
) -> ManualMetadata:
    """Returns metadata and a hierarchical table of contents for a specified manual."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.id == manual_id).first()
        if not manual:
            raise ToolError(f"Manual with id '{manual_id}' not found.")

        bookmarks = (
            db.query(Bookmark)
            .filter(Bookmark.manual_id == manual.id)
            .order_by(Bookmark.ordering)
            .all()
        )
        table_of_contents = _build_toc(bookmarks)

        manual_data = {
            "id": manual.id,
            "file_name": manual.file_name,
            "document_title": manual.document_title,
            "file_hash": manual.file_hash,
            "table_of_contents": table_of_contents,
        }
        return ManualMetadata.model_validate(manual_data)
    except Exception as e:
        logger.error(f"Error fetching metadata for manual_id '{manual_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


def _figure_ref(figure: Figure) -> FigureRef:
    """Builds the lightweight figure reference embedded in tool responses."""
    return FigureRef(
        id=figure.id,
        page=figure.page,
        caption=figure.caption,
        description=figure.description,
        bookmark_id=figure.bookmark_id,
    )


def _load_figures(db: Session, figure_ids: list[str]) -> dict[str, Figure]:
    """Loads the given figures in a single query, keyed by figure id."""
    if not figure_ids:
        return {}
    figures = db.scalars(select(Figure).where(Figure.id.in_(figure_ids))).all()
    return {figure.id: figure for figure in figures}


def _get_descendant_bookmark_ids(
    manual_id: str, bookmark_id: str, db: Session
) -> List[str]:
    """Retrieves the list of bookmark IDs for the given bookmark and all its descendants."""
    target_bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
    if not target_bookmark:
        raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")

    if target_bookmark.manual_id != manual_id:
        raise ToolError(
            f"Bookmark '{bookmark_id}' does not belong to manual '{manual_id}'."
        )

    # Efficiently find descendants.
    # Since we have 'ordering' and 'level', descendants follow immediately
    # and have level > target.level.
    # We stop when we hit a bookmark with level <= target.level.

    # Get all subsequent bookmarks for this manual
    subsequent_bookmarks = (
        db.query(Bookmark)
        .filter(
            Bookmark.manual_id == manual_id,
            Bookmark.ordering >= target_bookmark.ordering,
        )
        .order_by(Bookmark.ordering)
        .all()
    )

    descendant_ids = []
    # The first one is the target itself
    for bm in subsequent_bookmarks:
        if bm.id == bookmark_id:
            descendant_ids.append(bm.id)
            continue

        if bm.level > target_bookmark.level:
            descendant_ids.append(bm.id)
        else:
            # We reached a sibling or parent (level <= target), so we stop
            break

    return descendant_ids


def _chunk_ids_in_bookmarks(collection, manual_id: str, bookmark_ids: list[str]) -> set:
    """Chunk ids under a bookmark subtree, for filtering lexical hits.

    The FTS index stores the manual but not the bookmark, so a search narrowed
    to a section has to intersect its results with the chunks Chroma says are
    in it.
    """
    if not bookmark_ids:
        return set()
    got = collection.get(
        where={
            "$and": [
                {"manual_id": manual_id},
                {"bookmark_id": {"$in": list(bookmark_ids)}},
            ]
        },
        include=[],
    )
    return set(got["ids"])


def _fetch_in_order(collection, ids: list[str]) -> dict:
    """Reads chunks by id and returns them shaped like a `query()` result."""
    if not ids:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    at = {cid: n for n, cid in enumerate(got["ids"])}
    keep = [i for i in ids if i in at]
    return {
        "ids": [keep],
        "documents": [[got["documents"][at[i]] for i in keep]],
        "metadatas": [[got["metadatas"][at[i]] for i in keep]],
    }


@app.tool(
    name="get_markdown_content",
    description="""Fetches the Markdown content for a specific bookmark (section) within
      a manual using the Vector DB. This returns the pre-processed text chunks associated
      with the bookmark and its sub-sections.

    Figures (diagrams, drawings, screenshots) appear in the Markdown as a
    `[Figure: <figure_id> (page N)]` marker followed by the figure's caption,
    labels and description, and are also listed in the `figures` field in
    document order. Pass a figure id to `get_figure` to obtain the image itself.

    Workflow Example:
    1. Get a `bookmark_id` from the `table_of_contents` provided by
      `get_manual_metadata()`.
    2. Call `get_markdown_content(bookmark_id=...)` to get the content.
    3. Call `get_figure(figure_id=...)` for any figure you need to look at.""",
    tags={"manual", "content", "markdown"},
    annotations={"readOnlyHint": True},
)
def get_markdown_content(
    bookmark_id: Annotated[
        str,
        Field(
            description="""The unique ID of the bookmark, 
            obtained from `get_manual_metadata`."""
        ),
    ],
) -> MarkdownContent:
    """Returns the Markdown content for a specific bookmark from the Vector DB."""
    collection = _require_collection()

    db: Session = SessionLocal()
    try:
        # Resolve bookmark and manual
        bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
        if not bookmark:
            raise ToolError(f"Bookmark with id '{bookmark_id}' not found.")

        manual_id = bookmark.manual_id

        # Get all relevant bookmark IDs (hierarchical)
        target_bookmark_ids = _get_descendant_bookmark_ids(manual_id, bookmark_id, db)

        # Query ChromaDB
        # We want chunks where manual_id matches AND bookmark_id is in our list

        # Chroma where clause:
        # {"$and": [{"manual_id": manual_id}, {"bookmark_id": {"$in": target_bookmark_ids}}]}

        results = collection.get(
            where={
                "$and": [
                    {"manual_id": manual_id},
                    {"bookmark_id": {"$in": target_bookmark_ids}},
                ]
            },
            include=["documents", "metadatas"],
        )

        # results['documents'] is a list of strings
        # results['ids'] is a list of IDs.
        # 'get' does not guarantee an order, so chunks are sorted by their
        # chunk_index metadata, falling back to the trailing index of the
        # legacy "<manual_id>_<index>" chunk ids.
        combined = []
        if results["ids"] and results["documents"] and results["metadatas"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]

                idx = 0
                if "chunk_index" in meta:
                    idx = meta["chunk_index"]
                else:
                    # Legacy fallback
                    parts = doc_id.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        idx = int(parts[1])

                combined.append((idx, results["documents"][i], meta))

        # Sort by index
        combined.sort(key=lambda x: x[0])

        figure_ids = [
            str(meta["figure_id"])
            for _, _, meta in combined
            if meta.get("type") == "figure" and meta.get("figure_id")
        ]
        figures_by_id = _load_figures(db, figure_ids)

        # Text and table chunks overlap each other and are merged; a figure
        # chunk is self-contained and is kept as its own block, so the overlap
        # logic never glues it to a neighbour.
        blocks: List[str] = []
        pending_texts: List[str] = []
        figure_refs: List[FigureRef] = []

        for _, text, meta in combined:
            if meta.get("type") != "figure":
                pending_texts.append(text)
                continue

            if pending_texts:
                blocks.append(_merge_chunks(pending_texts))
                pending_texts = []

            figure_id = meta.get("figure_id")
            page = meta.get("page")
            page_label = str(int(page)) if isinstance(page, (int, float)) else str(page)
            if figure_id:
                header = f"[Figure: {figure_id} (page {page_label})]"
            else:
                header = f"[Figure (page {page_label})]"
            blocks.append(f"{header}\n\n{text}")

            if figure_id:
                figure = figures_by_id.get(str(figure_id))
                if figure is not None:
                    figure_refs.append(_figure_ref(figure))
                else:
                    logger.warning(
                        f"Figure '{figure_id}' is referenced by a chunk of "
                        f"bookmark '{bookmark_id}' but missing from the database."
                    )

        if pending_texts:
            blocks.append(_merge_chunks(pending_texts))

        final_content = "\n\n".join(blocks)

        return MarkdownContent(markdown_content=final_content, figures=figure_refs)

    except Exception as e:
        logger.exception(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


def _merge_chunks(chunks: List[str]) -> str:
    """
    Merges a list of text chunks, removing overlaps between adjacent chunks.
    Assumes chunks are sorted by their original sequence.
    """
    if not chunks:
        return ""
    
    merged = chunks[0]
    
    for next_chunk in chunks[1:]:
        # Find overlap between end of merged and start of next_chunk
        # Try to find the longest suffix of 'merged' that matches prefix of 'next_chunk'
        # We limit search to a reasonable window (e.g., slightly larger than chunk_overlap)
        
        overlap_len = 0
        max_overlap_search = settings.CHUNK_OVERLAP + settings.CHUNK_OVERLAP_SEARCH_MARGIN # Should cover chunk_overlap + margin
        
        # Search window in merged (last N chars)
        search_start_idx = max(0, len(merged) - max_overlap_search)
        suffix_window = merged[search_start_idx:]
        
        # Iterate over possible overlap lengths
        # Optimized: checking logical overlaps
        # It's cleaner to check if next_chunk starts with a suffix of merged
        for length in range(min(len(suffix_window), len(next_chunk)), 0, -1):
            if suffix_window.endswith(next_chunk[:length]):
                overlap_len = length
                break
        
        if overlap_len > 0:
            merged += next_chunk[overlap_len:]
        else:
            # No overlap detected. Likely a section break or distinct block.
            # Add separator.
            merged += "\n\n" + next_chunk

    return merged




@app.tool(
    name="search_manual",
    description="""Searches for a query string within a specific manual using semantic search.
    Returns the top matching chunks.

    Optionally, a `bookmark_id` can be provided to restrict the search to a specific
    section of the manual (including subsections).

    Every result reports its `chunk_type` ("text", "table" or "figure"). A hit
    with `chunk_type` "figure" also carries a `figure` object whose `id` can be
    passed to `get_figure` to retrieve the image itself; its `context` is the
    figure's caption, labels and description.

    Workflow Example:
    1. Call `search_manual(manual_id=..., query="...")` to find occurrences.
    2. Call `get_figure(figure_id=...)` for a hit whose `chunk_type` is "figure".
    """,
    tags={"manual", "search"},
    annotations={"readOnlyHint": True},
)
def search_manual(
    manual_id: Annotated[
        str,
        Field(description="The unique ID of the manual to search."),
    ],
    query: Annotated[
        str,
        Field(description="The text to search for."),
    ],
    bookmark_id: Annotated[
        Optional[str],
        Field(
            description="Optional bookmark ID to restrict search to a specific section."
        ),
    ] = None,
) -> SearchResult:
    """Searches for text in a manual and returns matches with context and hierarchy."""
    collection = _require_collection()

    if app_state.embedder is None:
        raise ToolError("Embedding model is not initialized.")

    embedder = app_state.embedder

    db: Session = SessionLocal()
    try:
        where_clause = {"manual_id": manual_id}

        if bookmark_id:
            # Hierarchical filter
            target_ids = _get_descendant_bookmark_ids(manual_id, bookmark_id, db)
            where_clause = {
                "$and": [{"manual_id": manual_id}, {"bookmark_id": {"$in": target_ids}}]
            }

        # Dense and lexical retrieval, fused by rank.
        #
        # Dense alone cannot find an identifier: measured against an *exact*
        # scan of the vectors, the top 5 for "what does message IEF450I mean"
        # held no chunk containing that string, though 13 chunks do. BM25 finds
        # them, and finds nothing at all for a question with no indexable term
        # -- a purely Japanese one against this English corpus -- which is why
        # the two are fused by rank rather than by score. An empty lexical list
        # simply leaves the dense ranking untouched.
        query_vec = [embedder.embed_query(query)]
        dense = collection.query(
            query_embeddings=query_vec,
            n_results=lexical.DENSE_CANDIDATES,
            where=where_clause,
        )
        dense_ids = dense["ids"][0] if dense["ids"] else []

        lexical_ids = lexical.search(
            lexical.sqlite_connection(db),
            query,
            limit=lexical.LEXICAL_CANDIDATES,
            manual_id=manual_id,
        )
        if bookmark_id:
            # The dense side got this through `where`; the lexical index does
            # not carry the bookmark, so it is filtered against the same set.
            allowed = set(dense_ids)
            allowed.update(
                _chunk_ids_in_bookmarks(collection, manual_id, target_ids)
            )
            lexical_ids = [i for i in lexical_ids if i in allowed]

        ordered = lexical.fuse_dense_and_lexical(dense_ids, lexical_ids)[:5]

        # One fetch for whatever the fusion chose, in that order.
        results = _fetch_in_order(collection, ordered)

        # results is a dict with lists of lists (for documents, metadatas, etc.)
        # Structure: {'ids': [['id1', ...]], 'metadatas': [[{...}, ...]], 'documents': [['text', ...]]}

        search_result_items = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            # Pre-fetch bookmarks for hierarchy reconstruction
            # We can't easily pre-fetch just the parents needed without knowing them.
            # But we can cache the manual's bookmarks map.
            all_bookmarks = (
                db.scalars(select(Bookmark).where(Bookmark.manual_id == manual_id)).all()
            )
            bookmark_map = {bm.id: bm for bm in all_bookmarks}

            # Figures referenced by the hits, resolved in a single query.
            figures_by_id = _load_figures(
                db, [str(m["figure_id"]) for m in metas if m.get("figure_id")]
            )

            for i, chunk_id in enumerate(ids):
                text = docs[i]
                meta = metas[i]

                # Get Bookmark Info
                chunk_bm_id = meta.get("bookmark_id")

                # Build hierarchy path
                bookmark_node_list = []

                if chunk_bm_id and chunk_bm_id in bookmark_map:
                    temp_bm = bookmark_map[chunk_bm_id]
                    path_nodes = []
                    while temp_bm:
                        path_nodes.append(temp_bm)
                        if temp_bm.parent_id:
                            temp_bm = bookmark_map.get(temp_bm.parent_id)
                        else:
                            temp_bm = None
                    path_nodes.reverse()

                    for node in path_nodes:
                        bookmark_node_list.append(
                            BookmarkNode(
                                id=node.id,
                                title=node.title,
                                page=node.page_num,
                                children=[],
                            )
                        )

                figure_ref = None
                figure_id = meta.get("figure_id")
                if figure_id:
                    figure = figures_by_id.get(str(figure_id))
                    if figure is not None:
                        figure_ref = _figure_ref(figure)
                    else:
                        logger.warning(
                            f"Chunk '{chunk_id}' references figure "
                            f"'{figure_id}', which is missing from the database."
                        )

                search_result_items.append(
                    SearchResultItem(
                        bookmarks=bookmark_node_list,
                        context=text,
                        manual_id=manual_id,
                        bookmark_id=chunk_bm_id,
                        chunk_type=str(meta.get("type", "text")),
                        figure=figure_ref,
                    )
                )

        return SearchResult(
            manual_id=manual_id, query=query, results=search_result_items
        )

    except Exception as e:
        logger.error(f"Error searching manual '{manual_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


@app.tool(
    name="get_figure",
    description="""Returns the image of a figure (diagram, drawing, screenshot)
    stored from a manual, together with its metadata.
    Figure ids come from `search_manual` results whose `chunk_type` is "figure"
    (field `figure.id`) and from the `figures` list of `get_markdown_content`.
    The response contains the PNG image and a JSON text block with the figure's
    manual_id, bookmark_id, page, caption, labels, description and size.""",
    tags={"manual", "figure"},
    annotations={"readOnlyHint": True},
    # Required: fastmcp would otherwise try to serialize the Image object as
    # structured content, which fails.
    output_schema=None,
)
def get_figure(
    figure_id: Annotated[str, Field(description="The unique ID of the figure.")],
):
    """Returns the PNG image of a figure plus its metadata as JSON."""
    db: Session = SessionLocal()
    try:
        figure = db.get(Figure, figure_id)
        if not figure:
            raise ToolError(f"Figure with id '{figure_id}' not found.")

        info = FigureInfo(
            id=figure.id,
            page=figure.page,
            caption=figure.caption,
            description=figure.description,
            bookmark_id=figure.bookmark_id,
            manual_id=figure.manual_id,
            labels=figure.labels,
            width=figure.width,
            height=figure.height,
            mime_type=figure.mime_type or "image/png",
        )
        return [
            Image(data=figure.image, format="png"),
            json.dumps(info.model_dump()),
        ]
    except Exception as e:
        logger.error(f"Error fetching figure '{figure_id}': {e}")
        raise ToolError(e)
    finally:
        db.close()


if __name__ == "__main__":
    app.run(transport="http", host=settings.HOST, port=settings.PORT)
