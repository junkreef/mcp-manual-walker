"""What the two front ends have in common.

The MCP tools in `main.py` and the REST endpoints in `rest_api.py` answer the
same five questions -- what is in the library, what is in a manual, what is in
a section, what matches a query and what does a figure look like -- and there
is no reason for either to have its own idea of the answer. Everything that
touches the databases lives here; the front ends only translate.

Two differences from the MCP tools this was lifted out of are deliberate:

* **Search is corpus-wide by default.** Both front ends accept a
  `manual_path` pattern, where `*` means the corpus and a path returned by
  `list_manuals` names one manual. `bookmark_id` can narrow either scope to a
  section and its descendants.
* **Failures are `ServiceError`, not `ToolError`.** A layer that raises
  `ToolError` cannot be served over HTTP without every 404 arriving as a 500,
  so the reason is carried in `kind` and each front end renders it: MCP as a
  `ToolError`, REST as a status code.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import List, Literal, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import lexical
from .config import settings
from .database import SessionLocal, init_db
from .embeddings import Embedder, check_embedding_model, get_embedder
from .models import Bookmark, Figure, Manual
from .schemas import (
    BookmarkNode,
    DirectoryEntry,
    FigureInfo,
    FigureRef,
    ManualMetadata,
    ManualRef,
    MarkdownContent,
    SearchHit,
)
from .vector_store import ChunkFilter, VectorStore, open_store

logger = logging.getLogger(__name__)

# Results returned when a caller does not say how many it wants. Five is what
# the MCP tool has always returned, and it is about what fits in a context
# window alongside a question.
DEFAULT_SEARCH_LIMIT = 5

# Upper bound a caller may ask for. Each result carries a whole chunk, so a
# large limit is a large response; this stops one request from reading a
# meaningful fraction of the corpus back out.
MAX_SEARCH_LIMIT = 50

ErrorKind = Literal["not_found", "unavailable", "invalid", "internal"]

# Guards every write to `app_state`, because two servers share it: the MCP
# server on the main thread and the REST API on its own. Without it a request
# arriving while the store is being attached sees the half-built state that
# `init_vector_store` clears on its way in, and -- worse -- can start loading a
# second copy of the embedding model concurrently with the first. That is
# 1.11 GiB of weights twice under the local backend, which is an OOM kill on a
# memory-capped container rather than a slow start.
#
# Reentrant because `ensure_initialized` holds it across `init_vector_store`.
_state_lock = threading.RLock()

# Whether this process has already opened the databases. Both servers need it
# done before they answer anything, and whichever gets there first should be
# the only one to pay for it.
_initialized = False


class ServiceError(Exception):
    """Something the caller asked for cannot be given, and why.

    `kind` exists so the REST layer can choose a status code without parsing
    the message: a manual that does not exist is a 404, a vector store that has
    not arrived yet is a 503, and the two are not the same problem.
    """

    def __init__(self, message: str, *, kind: ErrorKind = "internal") -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind


class AppState:
    def __init__(self) -> None:
        self.store: Optional[VectorStore] = None
        self.embedder: Optional[Embedder] = None
        # Reason why the vector store could not be used, surfaced by the tools.
        self.init_error: Optional[str] = None


app_state = AppState()


def ensure_initialized() -> None:
    """Opens the databases and attaches the store, once per process.

    Both servers call this before they serve anything. The REST API cannot
    simply rely on the MCP server's lifespan to have run: it is started first,
    from `main.serve`, and until this has happened `SessionLocal` is bound to
    no engine at all, so every request touching the relational database fails
    with an error about a missing bind rather than with anything a caller could
    act on.
    """
    global _initialized
    with _state_lock:
        if _initialized:
            return

        logger.info("Initializing application...")
        # Ensure all necessary directories exist before initializing the database
        settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.PDF_ROOT_DIR.mkdir(parents=True, exist_ok=True)
        settings.CHROMADB_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing database...")
        init_db()

        init_vector_store()
        _initialized = True


def init_vector_store() -> None:
    """
    Connects app_state to the persisted vector collection.

    Kept out of the lifespan so it can be re-run (the test suite reuses a single
    server object). A failure is recorded instead of raised, so the server still
    starts and every tool can explain why the vector store is unusable.
    """
    logger.info(f"Initializing the {settings.VECTOR_BACKEND} vector store...")
    with _state_lock:
        app_state.store = None
        app_state.embedder = None
        app_state.init_error = None

        try:
            store = open_store()

            # Load the embedding model (the same one the builder used)
            embedder = get_embedder()

            # Vectors built with another model are not comparable to ours.
            check_embedding_model(store.embedding_model, settings.EMBEDDING_MODEL)

            app_state.embedder = embedder
            app_state.store = store

        except Exception as e:
            logger.error(f"Failed to initialize the vector store: {e}")
            app_state.init_error = str(e)


def _reopen_store() -> bool:
    """Tries again to attach to a store that was not there at startup.

    A server started before its database exists is the normal case rather than
    an error: `docker compose up` brings this up next to an empty Qdrant, and
    the corpus arrives afterwards through `db_manager import`. Without this the
    server would answer every request with the same stale complaint until
    somebody restarted it.

    The embedding model is not reloaded. It is the expensive half of startup --
    1.11 GiB of weights under the local backend -- and it has nothing to do
    with whether the collection has appeared yet.

    Called with `_state_lock` held, so two requests arriving together on the
    two servers cannot both decide to load the model.
    """
    try:
        store = open_store()
        check_embedding_model(store.embedding_model, settings.EMBEDDING_MODEL)
    except Exception as e:  # noqa: BLE001 - reported through the caller below
        app_state.init_error = str(e)
        return False

    if app_state.embedder is None:
        app_state.embedder = get_embedder()
    app_state.store = store
    app_state.init_error = None
    logger.info("The vector store is now available.")
    return True


def require_store() -> VectorStore:
    """Returns the vector store, or raises explaining why it is gone."""
    store = app_state.store
    if store is not None:
        return store

    with _state_lock:
        # Another thread may have attached one while this one waited.
        if app_state.store is None and not _reopen_store():
            message = "Vector database is not initialized."
            if app_state.init_error:
                message = f"{message} {app_state.init_error}"
            raise ServiceError(message, kind="unavailable")
        assert app_state.store is not None
        return app_state.store


def require_embedder():
    """Returns the query embedder, or raises if the model never loaded."""
    if app_state.embedder is None:
        raise ServiceError("Embedding model is not initialized.", kind="unavailable")
    return app_state.embedder


def store_status() -> tuple[bool, Optional[str]]:
    """Whether the vector store is usable, and the reason when it is not.

    Unlike `require_store` this never raises: a health check wants to report
    the state, not to fail on it.
    """
    try:
        require_store()
    except ServiceError as e:
        return False, str(e)
    return True, None


def normalize_folder(folder: Optional[str]) -> str:
    """Turns a caller-supplied folder into a path prefix: `''` or `'a/b/'`.

    The trailing slash is what keeps the prefix match on a folder boundary, so
    that listing `zOS` cannot pick up a sibling folder named `zOS-legacy`.
    """
    if not folder:
        return ""
    cleaned = folder.replace("\\", "/").strip().strip("/")
    if not cleaned or cleaned == ".":
        return ""
    return f"{cleaned}/"


def resolve_manual_path(manual_path: str, db: Session) -> Optional[list[str]]:
    """Resolves a public manual path pattern to IDs used by the indexes.

    `*` deliberately crosses `/`, so `zOS/V3R1/*` includes manuals in nested
    folders as well as PDFs directly in V3R1. Returning None represents the
    whole corpus and lets both retrieval backends omit their filter entirely.
    """
    pattern = manual_path.replace("\\", "/").strip().strip("/")
    if not pattern or pattern == ".":
        raise ServiceError("manual_path must not be empty.", kind="invalid")
    if pattern == "*":
        return None

    expression = re.compile(re.escape(pattern).replace(r"\*", ".*") + r"\Z")
    manuals = db.execute(select(Manual.id, Manual.relative_path)).all()
    ids = [
        manual_id
        for manual_id, relative_path in manuals
        if expression.fullmatch(relative_path.replace("\\", "/"))
    ]
    if not ids:
        raise ServiceError(
            f"No manuals match manual_path '{manual_path}'.", kind="not_found"
        )
    return ids


def list_folder(folder: str = "") -> List[DirectoryEntry]:
    """Returns the manuals and subfolders directly inside the given folder."""
    prefix = normalize_folder(folder)
    db: Session = SessionLocal()
    try:
        manuals = db.query(Manual).order_by(Manual.relative_path).all()

        # Manual counts per immediate subdirectory, keyed by its name.
        subdirectories: dict[str, int] = {}
        files: list[DirectoryEntry] = []
        matched = False

        for m in manuals:
            # Paths written on Windows carry backslashes; the tool speaks `/`.
            relative_path = m.relative_path.replace("\\", "/")
            if prefix and not relative_path.startswith(prefix):
                continue
            matched = True

            remainder = relative_path[len(prefix) :]
            head, separator, _ = remainder.partition("/")
            if separator:
                subdirectories[head] = subdirectories.get(head, 0) + 1
            else:
                files.append(
                    DirectoryEntry(
                        type="manual",
                        name=m.file_name,
                        path=relative_path,
                        id=m.id,
                        document_title=m.document_title,
                    )
                )
    except ServiceError:
        raise
    except Exception as e:
        logger.error(f"Error listing folder '{folder}': {e}")
        raise ServiceError(str(e))
    finally:
        db.close()

    if prefix and not matched:
        raise ServiceError(
            f"No folder named '{folder}' exists in the manual library. "
            "Call `list_manuals()` without arguments to list the root, then "
            "follow the `path` of the directory entries.",
            kind="not_found",
        )

    directories = [
        DirectoryEntry(
            type="directory",
            name=name,
            path=f"{prefix}{name}",
            manual_count=count,
        )
        for name, count in sorted(subdirectories.items())
    ]
    return directories + sorted(files, key=lambda entry: entry.name)


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


def manual_metadata(manual_id: str) -> ManualMetadata:
    """Returns metadata and a hierarchical table of contents for a manual."""
    db: Session = SessionLocal()
    try:
        manual = db.query(Manual).filter(Manual.id == manual_id).first()
        if not manual:
            raise ServiceError(
                f"Manual with id '{manual_id}' not found.", kind="not_found"
            )

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
    except ServiceError:
        raise
    except Exception as e:
        logger.error(f"Error fetching metadata for manual_id '{manual_id}': {e}")
        raise ServiceError(str(e))
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


def _figure_info(figure: Figure) -> FigureInfo:
    """Builds the full description of a figure, image excluded."""
    return FigureInfo(
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


def _load_figures(db: Session, figure_ids: list[str]) -> dict[str, Figure]:
    """Loads the given figures in a single query, keyed by figure id."""
    if not figure_ids:
        return {}
    figures = db.scalars(select(Figure).where(Figure.id.in_(figure_ids))).all()
    return {figure.id: figure for figure in figures}


def get_figure(figure_id: str) -> tuple[bytes, FigureInfo]:
    """Returns a figure's PNG bytes and its metadata."""
    db: Session = SessionLocal()
    try:
        figure = db.get(Figure, figure_id)
        if not figure:
            raise ServiceError(
                f"Figure with id '{figure_id}' not found.", kind="not_found"
            )
        return figure.image, _figure_info(figure)
    except ServiceError:
        raise
    except Exception as e:
        logger.error(f"Error fetching figure '{figure_id}': {e}")
        raise ServiceError(str(e))
    finally:
        db.close()


def _get_descendant_bookmark_ids(
    manual_id: str, bookmark_id: str, db: Session
) -> List[str]:
    """Retrieves the bookmark IDs for the given bookmark and all its descendants."""
    target_bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
    if not target_bookmark:
        raise ServiceError(
            f"Bookmark with id '{bookmark_id}' not found.", kind="not_found"
        )

    if target_bookmark.manual_id != manual_id:
        raise ServiceError(
            f"Bookmark '{bookmark_id}' does not belong to manual '{manual_id}'.",
            kind="invalid",
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


def _chunk_ids_in_bookmarks(store, manual_id: str, bookmark_ids: list[str]) -> set:
    """Chunk ids under a bookmark subtree, for filtering lexical hits.

    The FTS index stores the manual but not the bookmark, so a search narrowed
    to a section has to intersect its results with the chunks the vector store
    says are in it.
    """
    if not bookmark_ids:
        return set()
    return {
        chunk.id
        for chunk in store.scroll(
            ChunkFilter(manual_ids=[manual_id], bookmark_ids=bookmark_ids),
            with_document=False,
        )
    }


def _bookmark_path(
    bookmark_map: dict[str, Bookmark], bookmark_id: Optional[str]
) -> list[BookmarkNode]:
    """The chain of bookmarks from the root of the manual down to this one."""
    if not bookmark_id or bookmark_id not in bookmark_map:
        return []

    path_nodes = []
    node: Optional[Bookmark] = bookmark_map[bookmark_id]
    while node:
        path_nodes.append(node)
        node = bookmark_map.get(node.parent_id) if node.parent_id else None
    path_nodes.reverse()

    return [
        BookmarkNode(id=n.id, title=n.title, page=n.page_num, children=[])
        for n in path_nodes
    ]


def _manual_refs(db: Session, manual_ids: Sequence[str]) -> dict[str, ManualRef]:
    """Loads the manuals behind a set of hits in a single query."""
    ids = [m for m in dict.fromkeys(manual_ids) if m]
    if not ids:
        return {}
    manuals = db.scalars(select(Manual).where(Manual.id.in_(ids))).all()
    return {
        m.id: ManualRef(
            id=m.id,
            file_name=m.file_name,
            document_title=m.document_title,
            relative_path=m.relative_path.replace("\\", "/"),
        )
        for m in manuals
    }


def search(
    query: str,
    *,
    manual_path: str = "*",
    bookmark_id: Optional[str] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> List[SearchHit]:
    """Hybrid search over manuals selected by path, optionally within a section.

    `manual_path` is either the `path` returned by `list_manuals`, a pattern
    containing `*`, or `*` for the whole corpus.
    """
    if not query or not query.strip():
        raise ServiceError("The search query is empty.", kind="invalid")
    if limit < 1 or limit > MAX_SEARCH_LIMIT:
        raise ServiceError(
            f"limit must be between 1 and {MAX_SEARCH_LIMIT}.", kind="invalid"
        )

    store = require_store()
    embedder = require_embedder()

    db: Session = SessionLocal()
    try:
        target_ids: list[str] = []
        selected_manual_ids = resolve_manual_path(manual_path, db)
        effective_manual_ids = selected_manual_ids
        bookmark_manual_id: Optional[str] = None

        if bookmark_id:
            bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
            if not bookmark:
                raise ServiceError(
                    f"Bookmark with id '{bookmark_id}' not found.", kind="not_found"
                )
            bookmark_manual_id = bookmark.manual_id
            if (
                selected_manual_ids is not None
                and bookmark_manual_id not in selected_manual_ids
            ):
                raise ServiceError(
                    f"Bookmark '{bookmark_id}' is outside manual_path '{manual_path}'.",
                    kind="invalid",
                )
            target_ids = _get_descendant_bookmark_ids(
                bookmark_manual_id, bookmark_id, db
            )
            effective_manual_ids = [bookmark_manual_id]
            where = ChunkFilter(
                manual_ids=effective_manual_ids, bookmark_ids=target_ids
            )
        elif selected_manual_ids is not None:
            where = ChunkFilter(manual_ids=selected_manual_ids)
        else:
            where = ChunkFilter()

        # Dense and lexical retrieval, fused by rank.
        #
        # Dense alone cannot find an identifier: measured against an *exact*
        # scan of the vectors, the top 5 for "what does message IEF450I mean"
        # held no chunk containing that string, though 13 chunks do. BM25 finds
        # them, and finds nothing at all for a question with no indexable term
        # -- a purely Japanese one against this English corpus -- which is why
        # the two are fused by rank rather than by score. An empty lexical list
        # simply leaves the dense ranking untouched.
        #
        # Both retrievers are asked for at least as many candidates as the
        # caller wants results, so a large `limit` widens the shortlist rather
        # than truncating a fixed one.
        dense_hits = store.search(
            embedder.embed_query(query),
            limit=max(lexical.DENSE_CANDIDATES, limit),
            where=where,
        )
        dense_ids = [hit.id for hit in dense_hits]
        dense_scores = {hit.id: hit.score for hit in dense_hits}

        lexical_ids = lexical.search(
            lexical.sqlite_connection(db),
            query,
            limit=max(lexical.LEXICAL_CANDIDATES, limit),
            manual_ids=effective_manual_ids,
        )
        if bookmark_id:
            # The dense side got this through the filter; the lexical index does
            # not carry the bookmark, so it is filtered against the same set.
            assert bookmark_manual_id is not None
            allowed = set(dense_ids)
            allowed.update(
                _chunk_ids_in_bookmarks(store, bookmark_manual_id, target_ids)
            )
            lexical_ids = [i for i in lexical_ids if i in allowed]

        ordered = lexical.fuse_dense_and_lexical(dense_ids, lexical_ids)[:limit]

        # One fetch for whatever the fusion chose, in that order.
        chunks = store.get(ordered)
        if not chunks:
            return []

        lexical_id_set = set(lexical_ids)
        dense_id_set = set(dense_ids)

        # Bookmarks are needed to rebuild the path to each hit. A corpus-wide
        # search can touch several manuals, so this loads the bookmarks of the
        # manuals that were actually hit rather than of one named in advance.
        hit_manual_ids = [
            str(chunk.metadata.get("manual_id"))
            for chunk in chunks
            if chunk.metadata.get("manual_id")
        ]
        bookmark_map: dict[str, Bookmark] = {}
        if hit_manual_ids:
            bookmark_map = {
                bm.id: bm
                for bm in db.scalars(
                    select(Bookmark).where(Bookmark.manual_id.in_(set(hit_manual_ids)))
                ).all()
            }
        manual_refs = _manual_refs(db, hit_manual_ids)

        # Figures referenced by the hits, resolved in a single query.
        figures_by_id = _load_figures(
            db,
            [
                str(chunk.metadata["figure_id"])
                for chunk in chunks
                if chunk.metadata.get("figure_id")
            ],
        )

        hits: List[SearchHit] = []
        for rank, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            chunk_manual_id = str(meta.get("manual_id") or bookmark_manual_id or "")
            chunk_bm_id = meta.get("bookmark_id")
            manual_ref = manual_refs.get(chunk_manual_id)

            figure_ref = None
            figure_id = meta.get("figure_id")
            if figure_id:
                figure = figures_by_id.get(str(figure_id))
                if figure is not None:
                    figure_ref = _figure_ref(figure)
                else:
                    logger.warning(
                        f"Chunk '{chunk.id}' references figure "
                        f"'{figure_id}', which is missing from the database."
                    )

            page = meta.get("page")

            retrieval: Literal["dense", "lexical", "both"]
            if chunk.id in dense_id_set and chunk.id in lexical_id_set:
                retrieval = "both"
            elif chunk.id in lexical_id_set:
                retrieval = "lexical"
            else:
                retrieval = "dense"

            hits.append(
                SearchHit(
                    bookmarks=_bookmark_path(bookmark_map, chunk_bm_id),
                    context=chunk.document or "",
                    manual_id=chunk_manual_id,
                    manual_path=manual_ref.relative_path if manual_ref else "",
                    bookmark_id=chunk_bm_id,
                    chunk_type=str(meta.get("type", "text")),
                    figure=figure_ref,
                    chunk_id=chunk.id,
                    rank=rank,
                    # A similarity only exists for a chunk the vector search
                    # returned; one found by BM25 alone was never scored
                    # against the query vector, and inventing a number for it
                    # would be worse than saying nothing.
                    score=dense_scores.get(chunk.id),
                    retrieval=retrieval,
                    page=int(page) if isinstance(page, (int, float)) else None,
                    manual=manual_refs.get(chunk_manual_id),
                )
            )

        return hits

    except ServiceError:
        raise
    except Exception as e:
        logger.error(f"Error searching for '{query}': {e}")
        raise ServiceError(str(e))
    finally:
        db.close()


def markdown_content(bookmark_id: str) -> MarkdownContent:
    """Returns the Markdown content for a specific bookmark from the Vector DB."""
    store = require_store()

    db: Session = SessionLocal()
    try:
        # Resolve bookmark and manual
        bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
        if not bookmark:
            raise ServiceError(
                f"Bookmark with id '{bookmark_id}' not found.", kind="not_found"
            )

        manual_id = bookmark.manual_id

        # Get all relevant bookmark IDs (hierarchical)
        target_bookmark_ids = _get_descendant_bookmark_ids(manual_id, bookmark_id, db)

        # Every chunk of this manual that belongs to the section or one of its
        # subsections.
        chunks = list(
            store.scroll(
                ChunkFilter(manual_ids=[manual_id], bookmark_ids=target_bookmark_ids)
            )
        )

        # The store does not promise an order, so chunks are sorted by their
        # chunk_index metadata, falling back to the trailing index of the
        # legacy "<manual_id>_<index>" chunk ids.
        combined = []
        for chunk in chunks:
            meta = chunk.metadata

            idx = 0
            if "chunk_index" in meta:
                idx = meta["chunk_index"]
            else:
                # Legacy fallback
                parts = chunk.id.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1])

            combined.append((idx, chunk.document, meta))

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
                # A chunk with no stored text is not expected, but merging one
                # would fail on a None rather than say so.
                pending_texts.append(text or "")
                continue

            if pending_texts:
                blocks.append(merge_chunks(pending_texts))
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
            blocks.append(merge_chunks(pending_texts))

        final_content = "\n\n".join(blocks)

        return MarkdownContent(markdown_content=final_content, figures=figure_refs)

    except ServiceError:
        raise
    except Exception as e:
        logger.exception(f"Error getting content for bookmark_id '{bookmark_id}': {e}")
        raise ServiceError(str(e))
    finally:
        db.close()


def merge_chunks(chunks: List[str]) -> str:
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
        # We limit search to a reasonable window (slightly larger than chunk_overlap)

        overlap_len = 0
        # Should cover chunk_overlap + margin
        max_overlap_search = (
            settings.CHUNK_OVERLAP + settings.CHUNK_OVERLAP_SEARCH_MARGIN
        )

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
