import contextlib
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
    feed it left-padded inputs and the right prompt. The prefixes default to
    the model's own stored prompts (`model.prompts`); an explicit prefix in
    settings overrides that.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        query_prefix: Optional[str],
        document_prefix: Optional[str],
        max_seq_length: int,
        batch_size: int,
        dtype: str = "auto",
        token_budget: int = 0,
    ):
        # Imported here so this module stays importable without torch installed.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._token_budget = token_budget
        self._device = device
        # "auto" has already been resolved by the caller; anything that is not
        # plain CPU is worth moving off the device between uses.
        self._is_accelerated = not str(device).startswith("cpu")

        # The dtype is always stated rather than left to the library default:
        # transformers 4.x resolved an unset dtype to float32 and 5.x resolves
        # it to "auto", so the footprint silently halved on an upgrade.
        self.model = SentenceTransformer(
            model_name, device=device, model_kwargs={"dtype": dtype}
        )

        # Left padding is required for last-token pooling: with right padding
        # the final position of a short input would be a pad token. This is
        # set directly on the tokenizer rather than via the constructor's
        # `tokenizer_kwargs` because that argument is deprecated in
        # sentence-transformers 6 (renamed to `processor_kwargs`, which does
        # not exist in 5.x); setting the attribute works on both.
        self.model.tokenizer.padding_side = "left"

        self.model.max_seq_length = max_seq_length

        self._query_prefix = self._resolve_prefix(query_prefix, "query")
        self._document_prefix = self._resolve_prefix(document_prefix, "document")
        logger.info("Resolved embedding query prefix: %r", self._query_prefix)
        logger.info("Resolved embedding document prefix: %r", self._document_prefix)

    def _resolve_prefix(self, configured: Optional[str], prompt_name: str) -> str:
        """
        Resolves a query/document prefix.

        An explicit setting (even an empty string) always wins; otherwise the
        model's own stored prompt for that name is used, if any.
        """
        if configured is not None:
            return configured
        prompts = getattr(self.model, "prompts", None) or {}
        return prompts.get(prompt_name, "")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def document_prefix(self) -> str:
        return self._document_prefix

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _token_length(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def plan_batches(self, lengths: list[int]) -> list[list[int]]:
        """Groups text indices into batches under a token budget.

        A batch is padded to its longest member, so its cost is
        ``len(batch) x longest``, not the sum of its lengths. Batching by row
        count therefore prices every batch at its worst member: measured on
        562 real chunks, a single 3836-token figure caption among 561 chunks
        averaging 298 tokens took the peak from 6.7 GB to 18.6 GB, because
        every batch it landed in was padded out to it.

        Budgeting tokens instead lets a batch of long texts be small and a
        batch of short ones be large. Inputs are visited longest-first so a
        long text starts a batch rather than joining one and forcing the rest
        to pad up to it; the returned batches are in that order, and callers
        put the results back in the original order themselves.
        """
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batches: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for index in order:
            length = max(1, lengths[index])
            candidate = max(longest, length)
            # A single text over budget still gets its own batch: truncation to
            # max_seq_length is the model's business, not this function's.
            if current and (
                (len(current) + 1) * candidate > self._token_budget
                or len(current) >= self._batch_size
            ):
                batches.append(current)
                current, longest = [], 0
                candidate = length
            current.append(index)
            longest = candidate
        if current:
            batches.append(current)
        return batches

    def _encode(
        self, texts: list[str], prompt: str, show_progress: bool = True
    ) -> list[list[float]]:
        if not texts:
            return []
        # An empty prefix is passed as None: Sentence Transformers would otherwise
        # tokenize "" to derive a prompt length, which is pointless at best.
        prompt = prompt or None
        if self._token_budget <= 0:
            vectors = self.model.encode(
                texts,
                prompt=prompt,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            return vectors.tolist()

        lengths = [self._token_length(text) for text in texts]
        results: list[Optional[list[float]]] = [None] * len(texts)
        for batch in self.plan_batches(lengths):
            vectors = self.model.encode(
                [texts[i] for i in batch],
                prompt=prompt,
                # Already grouped; one call per batch, so no further splitting.
                batch_size=len(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            for index, vector in zip(batch, vectors.tolist()):
                results[index] = vector
        return results  # type: ignore[return-value]

    def _empty_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - releasing is best effort
            logger.debug("Could not empty the device cache: %s", e)

    @contextlib.contextmanager
    def on_device(self):
        """Puts the model on the accelerator for the block, and takes it off.

        Two things keep a GPU occupied by an idle embedder, and both have to
        go. Torch's caching allocator holds the activation peak it reached in
        its own pool, and the weights themselves stay resident: measured in
        the builder's parent at 5114 MB long after its last batch, against
        1346 MB before its first, of which about 1.2 GB is the weights.

        On a GPU shared with the Docling workers that is not a cache, it is a
        reservation -- the worker that takes the freed slot finds the device
        still full. Moving the weights back and forth costs a fraction of a
        second against conversions measured in minutes.

        A CPU embedder has nothing to move, so this is a no-op there.
        """
        if not self._is_accelerated:
            yield
            return
        try:
            self.model.to(self._device)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not move the embedder to %s: %s", self._device, e)
        try:
            yield
        finally:
            try:
                self.model.to("cpu")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not move the embedder off the device: %s", e)
            self._empty_cache()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds passages for storage (no instruction prefix)."""
        return self._encode(list(texts), self._document_prefix)

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query (instruction prefix applied).

        No progress bar: one query is one batch, so the bar says nothing, and
        the search server would draw one per request. The builder keeps its
        bar, where thousands of chunks make it worth reading.
        """
        return self._encode([text], self._query_prefix, show_progress=False)[0]

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
        f"Loading embedding model {settings.EMBEDDING_MODEL} on device: {device} "
        f"(dtype={settings.EMBEDDING_DTYPE})"
    )
    try:
        return SentenceTransformerEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            device=device,
            query_prefix=settings.EMBEDDING_QUERY_PREFIX,
            document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
            max_seq_length=settings.EMBEDDING_MAX_SEQ_LENGTH,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            token_budget=settings.EMBEDDING_TOKEN_BUDGET,
            dtype=settings.EMBEDDING_DTYPE,
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
