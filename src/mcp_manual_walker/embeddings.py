import logging
from typing import Any, Optional

from mcp_manual_walker.config import settings

logger = logging.getLogger(__name__)

# Name of the single Chroma collection that holds every manual chunk.
COLLECTION_NAME = "manual_chunks"

# Collection metadata key recording which model produced the stored vectors.
EMBEDDING_MODEL_METADATA_KEY = "embedding_model"

# Hint printed whenever the embedding backend cannot be imported.
_INSTALL_HINT = (
    "install with `uv sync --extra cpu` (server) or `--extra builder` (GPU build)"
)


def _resolve_device(preferred: str) -> str:
    """
    Resolves the compute device for SentenceTransformers.

    Any explicit value ("cpu", "cuda", "cuda:1", "mps", ...) is passed through
    untouched. "auto" asks torch whether a CUDA device is usable and falls back
    to CPU when torch is unavailable.
    """
    if preferred.lower() != "auto":
        return preferred

    # Imported lazily and separately from sentence_transformers: the test suite
    # injects a fake sentence_transformers module without providing torch.
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SentenceTransformerEmbedder:
    """
    The single embedding path used by both the builder and the search server.

    Qwen3-Embedding is a decoder model: last-token pooling and L2 normalisation
    are part of its Sentence Transformers pipeline, so this class only has to
    feed it left-padded inputs and the right prompt. Queries carry an
    instruction prefix, documents carry none.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        query_prefix: str,
        document_prefix: str,
        max_seq_length: int,
        batch_size: int,
    ):
        # Imported here so this module stays importable without torch installed.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size

        # Left padding is required for last-token pooling: with right padding the
        # final position of a short input would be a pad token.
        self.model = SentenceTransformer(
            model_name,
            device=device,
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_seq_length

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        # An empty prefix is passed as None: Sentence Transformers would otherwise
        # tokenize "" to derive a prompt length, which is pointless at best.
        vectors = self.model.encode(
            texts,
            prompt=prompt or None,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds passages for storage (no instruction prefix)."""
        return self._encode(list(texts), self._document_prefix)

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query (instruction prefix applied)."""
        return self._encode([text], self._query_prefix)[0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Alias for embed_documents, for callers expecting a Chroma-style callable."""
        return self.embed_documents(input)


def get_embedder() -> Optional[SentenceTransformerEmbedder]:
    """
    Builds the embedder from the application settings.

    Returns None (after logging an actionable error) when sentence-transformers
    or its torch backend are not installed.
    """
    device = _resolve_device(settings.EMBEDDING_DEVICE)
    logger.info(
        f"Loading embedding model {settings.EMBEDDING_MODEL} on device: {device}"
    )
    try:
        return SentenceTransformerEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            device=device,
            query_prefix=settings.EMBEDDING_QUERY_PREFIX,
            document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
            max_seq_length=settings.EMBEDDING_MAX_SEQ_LENGTH,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    except ImportError:
        logger.error(f"sentence-transformers is not available: {_INSTALL_HINT}.")
        return None


def collection_metadata(embedder: SentenceTransformerEmbedder) -> dict[str, Any]:
    """Metadata stored on the Chroma collection when it is first created."""
    return {
        EMBEDDING_MODEL_METADATA_KEY: embedder.model_name,
        "embedding_dim": embedder.dimension,
        "hnsw:space": "cosine",
        "description": "Chunks from PDF manuals",
    }


def check_collection_model(collection: Any, expected_model: str) -> None:
    """
    Verifies that a Chroma collection was built with the expected model.

    Vectors from different models are not comparable, so a mismatch has to stop
    the caller instead of silently returning nonsense results.
    """
    metadata = getattr(collection, "metadata", None) or {}
    stored_model = metadata.get(EMBEDDING_MODEL_METADATA_KEY)

    if stored_model is None:
        raise RuntimeError(
            "The vector collection does not record an embedding model, so it "
            "predates this check and was almost certainly built with another "
            f"model; settings.EMBEDDING_MODEL is '{expected_model}'. "
            "Rebuild with `db_manager build --reset`."
        )

    if stored_model != expected_model:
        raise RuntimeError(
            f"The vector collection was built with '{stored_model}', but "
            f"settings.EMBEDDING_MODEL is '{expected_model}'. "
            "Rebuild with `db_manager build --reset`."
        )
