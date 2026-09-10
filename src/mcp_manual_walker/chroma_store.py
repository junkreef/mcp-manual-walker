"""The ChromaDB implementation of VectorStore.

Every Chroma-specific thing the application used to do inline lives here: the
`where` dictionaries, the nested lists a `query()` returns, the fact that
`get()` does not promise an order, the batch ceiling on `add()`, and the fact
that a cosine `distance` counts down where a similarity counts up.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any, Iterator, Optional, Sequence

# Imported at module scope, unlike in the rest of the package: this module *is*
# the Chroma backend and has no meaning without it. Laziness is preserved one
# level up, where `vector_store.open_store` imports this module only when the
# configured backend is Chroma -- so `db_manager watch` still starts without
# chromadb resident.
import chromadb

from mcp_manual_walker.config import settings
from mcp_manual_walker.embeddings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_METADATA_KEY,
    collection_metadata,
)
from mcp_manual_walker.vector_store import (
    DEFAULT_SCROLL_BATCH,
    DEFAULT_WRITE_BATCH,
    Chunk,
    ChunkFilter,
    Hit,
)

logger = logging.getLogger(__name__)


def _where(filter: ChunkFilter) -> Optional[dict]:
    """Renders a ChunkFilter as a Chroma `where` clause.

    Chroma rejects a one-element `$and`, so a single condition is emitted bare.
    """
    conditions: list[dict] = []
    if filter.manual_ids is not None:
        conditions.append({"manual_id": {"$in": list(filter.manual_ids)}})
    if filter.bookmark_ids is not None:
        conditions.append({"bookmark_id": {"$in": list(filter.bookmark_ids)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ChromaVectorStore:
    """VectorStore backed by a persistent local ChromaDB collection."""

    def __init__(self, client: Any, collection: Any):
        self._client = client
        self._collection = collection

    @classmethod
    def open(cls, embedder: Any = None, create: bool = False) -> "ChromaVectorStore":
        client = chromadb.PersistentClient(path=str(settings.CHROMADB_PATH.resolve()))
        if create:
            # No embedding function, ever: queries and documents are embedded by
            # the application so that the model, its prompt and its device are
            # decided in one place. The metadata is only honoured when the
            # collection is new -- get_or_create ignores it for an existing one,
            # which is why the caller still has to check the recorded model.
            metadata = collection_metadata(embedder) if embedder is not None else None
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                metadata=metadata,
            )
        else:
            collection = client.get_collection(name=COLLECTION_NAME)
        return cls(client, collection)

    @staticmethod
    def reset() -> None:
        """Deletes the whole store on disk, so the next open() starts clean."""
        path = settings.CHROMADB_PATH
        if path.exists():
            logger.warning(f"Resetting ChromaDB directory: {path}")
            shutil.rmtree(path)

    @property
    def embedding_model(self) -> Optional[str]:
        metadata = getattr(self._collection, "metadata", None) or {}
        return metadata.get(EMBEDDING_MODEL_METADATA_KEY)

    def add(self, chunks: Sequence[Chunk]) -> None:
        # Chroma rejects very large single batches, so inserts are sliced here
        # rather than at each call site.
        for start in range(0, len(chunks), DEFAULT_WRITE_BATCH):
            batch = chunks[start : start + DEFAULT_WRITE_BATCH]
            self._collection.add(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.document or "" for chunk in batch],
                metadatas=[dict(chunk.metadata) for chunk in batch],
                embeddings=[chunk.embedding for chunk in batch],
            )

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        got = self._collection.get(ids=list(ids), include=["documents", "metadatas"])
        at = {chunk_id: n for n, chunk_id in enumerate(got["ids"])}
        return [
            Chunk(
                id=chunk_id,
                document=got["documents"][at[chunk_id]],
                metadata=got["metadatas"][at[chunk_id]] or {},
            )
            for chunk_id in ids
            if chunk_id in at
        ]

    def scroll(
        self,
        where: ChunkFilter,
        *,
        with_document: bool = True,
        with_embedding: bool = False,
        batch_size: int = DEFAULT_SCROLL_BATCH,
    ) -> Iterator[Chunk]:
        if where.matches_nothing():
            return

        include = []
        if with_document:
            include.append("documents")
        if with_embedding:
            include.append("embeddings")
        # Metadata comes back whenever anything else does; asking for it
        # explicitly costs nothing and keeps the Chunk shape uniform.
        if include:
            include.append("metadatas")

        clause = _where(where)
        if clause is not None:
            # One filtered read. Deliberately not paged: a `$in` over several
            # manuals fails once the *matching chunks* exceed SQLite's
            # parameter limit, and the callers that filter already work one
            # manual (or one section) at a time.
            yield from self._decode(
                self._collection.get(where=clause, include=include),
                with_embedding,
            )
            return

        # The whole corpus, paged, because the two callers that ask for it walk
        # half a million chunks.
        offset = 0
        while True:
            got = self._collection.get(
                limit=batch_size, offset=offset, include=include
            )
            if not got["ids"]:
                return
            yield from self._decode(got, with_embedding)
            offset += len(got["ids"])

    @staticmethod
    def _decode(got: dict, with_embedding: bool) -> Iterator[Chunk]:
        documents = got.get("documents")
        metadatas = got.get("metadatas")
        embeddings = got.get("embeddings")
        for i, chunk_id in enumerate(got["ids"]):
            embedding = None
            if with_embedding and embeddings is not None and i < len(embeddings):
                embedding = embeddings[i]
                # Chroma hands vectors back as numpy arrays.
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
            yield Chunk(
                id=chunk_id,
                document=documents[i] if documents is not None else None,
                metadata=(metadatas[i] if metadatas is not None else None) or {},
                embedding=embedding,
            )

    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        where: Optional[ChunkFilter] = None,
    ) -> list[Hit]:
        if where is not None and where.matches_nothing():
            return []

        results = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=limit,
            where=_where(where) if where is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # The collections are built with "hnsw:space": "cosine", where
            # Chroma's distance is 1 - similarity. Every backend reports a
            # similarity, so a caller never has to know which way it counts.
            distance = distances[i] if i < len(distances) else None
            hits.append(
                Hit(
                    id=chunk_id,
                    score=1.0 - distance if distance is not None else 0.0,
                    document=documents[i] if i < len(documents) else None,
                    metadata=(metadatas[i] if i < len(metadatas) else None) or {},
                )
            )
        return hits

    def delete_manual(self, manual_id: str) -> None:
        self._collection.delete(where={"manual_id": manual_id})

    def count(self) -> int:
        return self._collection.count()

    def close(self) -> None:
        # PersistentClient owns a SQLite handle and an index cache; there is
        # nothing to close explicitly, and dropping the references is enough.
        self._collection = None
        self._client = None
