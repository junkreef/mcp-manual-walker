"""The Qdrant implementation of VectorStore.

Qdrant is here for one measured reason. Chroma keeps its vectors and its HNSW
graph in an anonymous hnswlib arena that has to stay resident; Qdrant keeps
both in memory-mapped files (`storage_type: InRamMmap`) and lets the kernel
decide what stays. On 684,398 chunks of 1024 dimensions that is the difference
between roughly 2.9 GB the process must have and roughly 350 MB it must have
plus 2.7 GB the operating system may reclaim.

Running that corpus in a container capped at 512 MB with swap disabled, a
filtered search still answered in 2.58 ms and a corpus-wide one in 5.02 ms.
That measures the process, not the machine: the host had memory to spare, so
the mmapped pages stayed in the global page cache. Somewhere genuinely short
of memory they would be evicted and re-read and the times would rise. What the
cap does establish is that nothing has to be anonymous memory, so nothing is
killed -- which is exactly what Chroma cannot say at this size.

Three things this backend has to do that Chroma's does not:

* **Point ids.** Qdrant accepts unsigned integers or UUIDs, and this
  application's chunk ids are strings ("<manual_id>_<n>"). A UUIDv5 over the
  chunk id is deterministic, so a chunk lands on the same point on every
  import, and the original id travels in the payload. The BM25 index in SQLite
  and the export archive therefore keep naming chunks exactly as they do now.

* **Payload indexes.** Every search this application issues is filtered by
  manual_id. Without a keyword index on it Qdrant cannot use the filter while
  traversing the graph.

* **The embedding model name.** Qdrant has no collection metadata, so the name
  is kept in the relational database (`vector_store.record_embedding_model`).

The measured configuration is the plain one. Moving vectors on disk or adding
scalar quantization was tried on the same corpus and made things worse on both
axes: `always_ram` int8 *added* 700 MB of unreclaimable anonymous memory that
the mmapped originals do not need, and filtered search went from 2.6 ms to
20.4 ms. The knobs are exposed because a corpus several times this one may
answer differently, but they default to off.

One difference from Chroma survives a round trip and is worth knowing about:
**Qdrant L2-normalizes the vectors of a cosine collection when it stores
them**, so an export taken from Qdrant returns unit vectors rather than the
exact numbers that went in. Measured on 3,273 real chunks whose stored norms
ranged 0.998047-1.003795 (bfloat16 rounding in the builder), the largest
per-dimension difference was 5.99e-04 as stored and 2.42e-08 once the source
was normalized too. Cosine ranking is scale-invariant, so nothing about
retrieval changes; a byte-for-byte archive comparison across backends will
notice, and should compare normalized vectors.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterator, Optional, Sequence

from qdrant_client import QdrantClient, models

from mcp_manual_walker.config import settings
from mcp_manual_walker.embeddings import HNSW_EF_CONSTRUCTION, HNSW_MAX_NEIGHBORS
from mcp_manual_walker.vector_store import (
    DEFAULT_SCROLL_BATCH,
    DEFAULT_WRITE_BATCH,
    Chunk,
    ChunkFilter,
    Hit,
    forget_embedding_model,
    record_embedding_model,
    recorded_embedding_model,
)

logger = logging.getLogger(__name__)

# Fixed namespace for chunk id -> point id. It cannot be random or generated:
# the mapping has to come out the same in every process and on every rebuild,
# or a re-import would write every chunk to a second point instead of
# overwriting the first.
CHUNK_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Payload keys that are not chunk metadata. They are stripped on the way out so
# that a chunk read back is byte-identical to the one written -- which the
# export archive depends on.
CHUNK_ID_FIELD = "chunk_id"
DOCUMENT_FIELD = "document"
_RESERVED = (CHUNK_ID_FIELD, DOCUMENT_FIELD)

# Fields the application filters on. Everything else is returned, never
# matched, and so needs no index.
INDEXED_FIELDS = ("manual_id", "bookmark_id")


def point_id(chunk_id: str) -> str:
    """The deterministic Qdrant point id for an application chunk id."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def _filter(where: Optional[ChunkFilter]) -> Optional[models.Filter]:
    """Renders a ChunkFilter as a Qdrant filter."""
    if where is None:
        return None
    conditions: list[models.Condition] = []
    if where.manual_id is not None:
        conditions.append(
            models.FieldCondition(
                key="manual_id", match=models.MatchValue(value=where.manual_id)
            )
        )
    if where.bookmark_ids is not None:
        conditions.append(
            models.FieldCondition(
                key="bookmark_id", match=models.MatchAny(any=list(where.bookmark_ids))
            )
        )
    return models.Filter(must=conditions) if conditions else None


def _manual_filter(manual_id: str) -> models.Filter:
    """The same filter, where the caller knows there is one."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="manual_id", match=models.MatchValue(value=manual_id)
            )
        ]
    )


_QUANTIZATION_KINDS = ("none", "int8", "binary")


def _quantization_kind() -> str:
    """The configured quantization, validated once for every caller.

    Validated here rather than only where a collection is created: a typo
    would otherwise pass silently at search time, where it would switch on
    rescoring against a quantized copy that does not exist.
    """
    kind = settings.QDRANT_QUANTIZATION.strip().lower() or "none"
    if kind not in _QUANTIZATION_KINDS:
        raise ValueError(
            f"QDRANT_QUANTIZATION is '{settings.QDRANT_QUANTIZATION}'; "
            "expected 'none', 'int8' or 'binary'."
        )
    return kind


def _quantization() -> Optional[Any]:
    kind = _quantization_kind()
    if kind == "int8":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        )
    if kind == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        )
    return None


def _search_params() -> models.SearchParams:
    quantization = None
    if _quantization_kind() != "none":
        # A quantized shortlist has to be rescored against the real vectors or
        # the ranking is the quantizer's, not the model's.
        quantization = models.QuantizationSearchParams(
            rescore=True, oversampling=settings.QDRANT_OVERSAMPLING
        )
    return models.SearchParams(
        hnsw_ef=settings.QDRANT_HNSW_EF, quantization=quantization
    )


class QdrantVectorStore:
    """VectorStore backed by a Qdrant collection."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int | None):
        self._client = client
        self._collection = collection
        # None until something needs the collection to exist. An import can be
        # asked to create a store without a usable embedding model, and the
        # first batch of vectors states the dimension just as well.
        self._dimension = dimension
        self._ready = False

    # --- lifecycle --------------------------------------------------------

    @staticmethod
    def _connect() -> QdrantClient:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=int(settings.QDRANT_TIMEOUT),
        )

    @classmethod
    def open(cls, embedder: Any = None, create: bool = False) -> "QdrantVectorStore":
        client = cls._connect()
        collection = settings.QDRANT_COLLECTION

        dimension = getattr(embedder, "dimension", None) if embedder else None
        store = cls(client, collection, dimension)

        if create:
            if dimension is not None:
                store._ensure_collection(dimension)
                if embedder is not None and recorded_embedding_model() is None:
                    # Only when nothing has claimed the store yet: one that
                    # already holds vectors keeps the name it was built with,
                    # so opening it under a different EMBEDDING_MODEL is caught
                    # rather than silently overwritten.
                    record_embedding_model(embedder.model_name)
            return store

        if not client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection '{collection}' does not exist at "
                f"{settings.QDRANT_URL}. Build it with `db_manager build`, or "
                "import an archive with `db_manager import`."
            )
        store._ready = True
        return store

    def _ensure_collection(self, dimension: int) -> None:
        if self._ready:
            return
        self._dimension = dimension

        if not self._client.collection_exists(self._collection):
            logger.info(
                f"Creating Qdrant collection '{self._collection}' "
                f"({dimension} dimensions, cosine)."
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=settings.QDRANT_ON_DISK_VECTORS,
                ),
                # The same graph parameters the Chroma collections are built
                # with, and for the same measured reason: at half a million
                # vectors the defaults leave recall dependent on the order the
                # vectors arrived in.
                hnsw_config=models.HnswConfigDiff(
                    m=HNSW_MAX_NEIGHBORS,
                    ef_construct=HNSW_EF_CONSTRUCTION,
                ),
                quantization_config=_quantization(),
                on_disk_payload=settings.QDRANT_ON_DISK_PAYLOAD,
            )
            for field in INDEXED_FIELDS:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        self._ready = True

    @staticmethod
    def reset() -> None:
        """Deletes the collection and the model name recorded beside it."""
        client = QdrantVectorStore._connect()
        collection = settings.QDRANT_COLLECTION
        try:
            if client.collection_exists(collection):
                logger.warning(f"Resetting Qdrant collection: {collection}")
                client.delete_collection(collection)
        finally:
            client.close()
        forget_embedding_model()

    @property
    def embedding_model(self) -> Optional[str]:
        return recorded_embedding_model()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None  # type: ignore[assignment]

    # --- writes -----------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        if not self._ready:
            first = next((c.embedding for c in chunks if c.embedding), None)
            if first is None:
                raise ValueError("Cannot create a Qdrant collection from chunks "
                                 "that carry no vectors.")
            self._ensure_collection(len(first))

        for start in range(0, len(chunks), DEFAULT_WRITE_BATCH):
            batch = chunks[start : start + DEFAULT_WRITE_BATCH]
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=point_id(chunk.id),
                        vector=list(chunk.embedding or []),
                        payload={
                            **dict(chunk.metadata),
                            CHUNK_ID_FIELD: chunk.id,
                            DOCUMENT_FIELD: chunk.document or "",
                        },
                    )
                    for chunk in batch
                ],
                # The caller's next action is another batch, not a read; the
                # write is confirmed by the collection going green afterwards.
                wait=False,
            )

    def delete_manual(self, manual_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=_manual_filter(manual_id)),
            wait=True,
        )

    # --- reads ------------------------------------------------------------

    @staticmethod
    def _to_chunk(payload: Optional[dict], vector: Any = None) -> Chunk:
        payload = dict(payload or {})
        chunk_id = payload.pop(CHUNK_ID_FIELD, None)
        document = payload.pop(DOCUMENT_FIELD, None)
        return Chunk(
            id=str(chunk_id) if chunk_id is not None else "",
            document=document,
            metadata=payload,
            embedding=list(vector) if vector is not None else None,
        )

    def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id(chunk_id) for chunk_id in ids],
            with_payload=True,
        )
        found = {}
        for record in records:
            chunk = self._to_chunk(record.payload)
            found[chunk.id] = chunk
        return [found[chunk_id] for chunk_id in ids if chunk_id in found]

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

        # The chunk id and the metadata are always needed; the text is the
        # largest thing in a payload and is left on the server when the caller
        # only wants ids.
        payload_selector: Any = True
        if not with_document:
            payload_selector = models.PayloadSelectorExclude(exclude=[DOCUMENT_FIELD])

        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=_filter(where),
                limit=batch_size,
                offset=offset,
                with_payload=payload_selector,
                with_vectors=with_embedding,
            )
            for record in records:
                yield self._to_chunk(record.payload, record.vector)
            if offset is None:
                return

    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        where: Optional[ChunkFilter] = None,
    ) -> list[Hit]:
        if where is not None and where.matches_nothing():
            return []

        response = self._client.query_points(
            collection_name=self._collection,
            query=list(embedding),
            limit=limit,
            query_filter=_filter(where),
            search_params=_search_params(),
            with_payload=True,
        )
        hits = []
        for point in response.points:
            chunk = self._to_chunk(point.payload)
            # Qdrant scores a cosine collection as a similarity already, which
            # is the direction Hit.score is defined in.
            hits.append(
                Hit(
                    id=chunk.id,
                    score=float(point.score),
                    document=chunk.document,
                    metadata=chunk.metadata,
                )
            )
        return hits

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection, exact=True
        ).count
