"""The interface every vector backend has to satisfy, and nothing more.

This is deliberately narrower than any one engine's API. It was derived by
cataloguing every call the application actually makes -- eight operations and
exactly three filter shapes -- rather than by abstracting over what Chroma or
Qdrant can do. An interface that admitted everything either engine offers would
be an interface neither could implement.

The filters the application uses are `manual_id in {...}`, `bookmark_id in
{...}`, and the two together, so `ChunkFilter` expresses those and refuses the
rest. Chroma renders it as a `where` dict, Qdrant as a `Filter(must=[...])`.

Two conventions the backends have to normalise, because they disagree:

* **Score direction.** Chroma returns a cosine *distance* (smaller is better),
  Qdrant a cosine *similarity* (larger is better). `Hit.score` is always a
  similarity, so a caller never has to know which engine answered.
* **Chunk identity.** The application's ids are strings ("<manual_id>_<n>").
  Chroma stores them as they are; Qdrant accepts only integers or UUIDs, so its
  backend maps each id to a deterministic UUIDv5 and keeps the original in the
  payload. Ids crossing this interface are always the application's own, which
  is what lets the BM25 index in SQLite and the export archive keep referring
  to chunks the way they already do.

One asymmetry is worth stating here rather than in either backend: **Qdrant has
no collection metadata**, so it has nowhere to put the embedding model name
that `embedding_model` returns and the server checks at startup. It keeps the
name in a small table in the application's own SQLite database instead (see
`record_embedding_model` below). The Chroma backend goes on reading it from the
collection metadata, which is where every existing database already has it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

# Chunks handed to a backend in one write. Chroma rejects very large single
# batches; Qdrant is happy with more but gains little from it.
DEFAULT_WRITE_BATCH = 1000

# Chunks read back in one page when walking the whole collection.
DEFAULT_SCROLL_BATCH = 5000


@dataclass(frozen=True)
class ChunkFilter:
    """The only three ways this application ever narrows a set of chunks.

    An empty filter matches everything. `bookmark_ids` is an explicit set
    rather than a subtree query because the hierarchy lives in SQLite, and the
    caller has already flattened it there.
    """

    manual_ids: Optional[Sequence[str]] = None
    bookmark_ids: Optional[Sequence[str]] = None

    def is_empty(self) -> bool:
        return self.manual_ids is None and self.bookmark_ids is None

    def matches_nothing(self) -> bool:
        """True when the filter cannot match, so a backend can skip the call.

        An empty explicit set is not "any": it means path or bookmark
        resolution found no eligible chunks, and it must match none.
        """
        return (
            self.manual_ids is not None
            and len(self.manual_ids) == 0
            or self.bookmark_ids is not None
            and len(self.bookmark_ids) == 0
        )


@dataclass
class Chunk:
    """One stored chunk: its id, its text, its metadata and its vector."""

    id: str
    document: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    embedding: Optional[Sequence[float]] = None


@dataclass
class Hit:
    """One search result. `score` is a similarity: larger is better."""

    id: str
    score: float
    document: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """What the builder, the search server and db_manager need from a backend."""

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Stores chunks, overwriting any that already exist under those ids."""

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        """Reads chunks by id, in the order asked for.

        Ids that are not present are dropped rather than reported: the caller
        is reconciling two indexes that can disagree (the BM25 index can name a
        chunk the vector store no longer holds) and wants what survives.
        """

    def scroll(
        self,
        where: ChunkFilter,
        *,
        with_document: bool = True,
        with_embedding: bool = False,
        batch_size: int = DEFAULT_SCROLL_BATCH,
    ) -> Iterator[Chunk]:
        """Iterates every chunk matching the filter, in no particular order.

        A stream rather than a list, because two of its three callers walk the
        whole corpus: the export writes half a million chunks to an archive and
        the lexical rebuild reads them all back. Chroma pages this with
        limit/offset and Qdrant with a point-id cursor, and neither should be
        the caller's problem.

        Order is explicitly not promised. Chunks that need to be in document
        order are sorted by their `chunk_index` metadata, which the caller
        already does.
        """

    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        where: Optional[ChunkFilter] = None,
    ) -> list[Hit]:
        """Nearest neighbours of an already-embedded query, best first.

        The query vector is always passed in. No backend is ever configured
        with an embedding function of its own: the application embeds queries
        itself so that the model, its prompt and its device are decided in one
        place.
        """

    def delete_manual(self, manual_id: str) -> None:
        """Removes every chunk belonging to one manual."""

    def count(self) -> int:
        """Total chunks held."""

    @property
    def embedding_model(self) -> Optional[str]:
        """The model recorded when the store was created, or None if unknown.

        Vectors from two models are not comparable, so the server checks this
        before serving. Chroma keeps it in the collection metadata; Qdrant has
        no such thing and its backend has to persist it somewhere itself.
        """

    def close(self) -> None:
        """Releases whatever the backend holds. Safe to call more than once."""


_EXPECTED = "expected 'chroma' or 'qdrant'"


def open_store(embedder: Any = None, create: bool = False) -> VectorStore:
    """Opens the configured backend.

    `embedder` is only consulted when a store is being created, to record which
    model its vectors came from; a reader passes nothing. Imported lazily so
    that a process which never touches vectors -- and an environment which has
    only one backend's client library installed -- does not pay for the other.
    """
    from mcp_manual_walker.config import settings

    backend = settings.VECTOR_BACKEND.strip().lower()
    if backend == "chroma":
        from mcp_manual_walker.chroma_store import ChromaVectorStore

        return ChromaVectorStore.open(embedder=embedder, create=create)
    if backend == "qdrant":
        from mcp_manual_walker.qdrant_store import QdrantVectorStore

        return QdrantVectorStore.open(embedder=embedder, create=create)

    raise ValueError(f"VECTOR_BACKEND is '{settings.VECTOR_BACKEND}'; {_EXPECTED}.")


def reset_store() -> None:
    """Discards the configured store's contents, so the next build starts clean."""
    from mcp_manual_walker.config import settings

    backend = settings.VECTOR_BACKEND.strip().lower()
    if backend == "chroma":
        from mcp_manual_walker.chroma_store import ChromaVectorStore

        ChromaVectorStore.reset()
        return
    if backend == "qdrant":
        from mcp_manual_walker.qdrant_store import QdrantVectorStore

        QdrantVectorStore.reset()
        return

    raise ValueError(f"VECTOR_BACKEND is '{settings.VECTOR_BACKEND}'; {_EXPECTED}.")


def batched(chunks: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Groups chunks into lists of at most `size`."""
    batch: list[Chunk] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# Key under which the embedding model name is kept in `vector_store_meta`.
EMBEDDING_MODEL_META_KEY = "embedding_model"


def _meta_session():
    """A session on the relational database, opening it if nothing else has.

    The vector store can be reached before `init_db` in a process that only
    wanted vectors, and a backend that keeps its model name here would
    otherwise fail with SQLAlchemy's "could not locate a bind", which says
    nothing about what went wrong. `init_db` is idempotent.
    """
    from mcp_manual_walker import database

    if database.engine is None:
        database.init_db()
    return database.SessionLocal()


def record_embedding_model(name: str) -> None:
    """Records which model produced the stored vectors, for backends that cannot.

    See models.VectorStoreMeta for why this lives in the relational database.
    Written only when a store is created, never on open: a store that already
    holds vectors keeps the name it was built with, so that opening it with a
    different EMBEDDING_MODEL is caught rather than silently overwritten.
    """
    from mcp_manual_walker.models import VectorStoreMeta

    with _meta_session() as session:
        row = session.get(VectorStoreMeta, EMBEDDING_MODEL_META_KEY)
        if row is None:
            session.add(VectorStoreMeta(key=EMBEDDING_MODEL_META_KEY, value=name))
        else:
            row.value = name
        session.commit()


def recorded_embedding_model() -> Optional[str]:
    """The recorded model name, or None if nothing has recorded one."""
    from mcp_manual_walker.models import VectorStoreMeta

    with _meta_session() as session:
        row = session.get(VectorStoreMeta, EMBEDDING_MODEL_META_KEY)
        return row.value if row is not None else None


def forget_embedding_model() -> None:
    """Drops the recorded name, so a reset store does not look like an old one."""
    from mcp_manual_walker.models import VectorStoreMeta

    with _meta_session() as session:
        row = session.get(VectorStoreMeta, EMBEDDING_MODEL_META_KEY)
        if row is not None:
            session.delete(row)
            session.commit()
