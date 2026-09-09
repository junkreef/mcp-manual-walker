import contextlib
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

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

# Prompts a remote endpoint cannot tell us about.
#
# With SentenceTransformers the prefixes come from the model's own
# config_sentence_transformers.json. An OpenAI-compatible endpoint exposes no
# such thing -- llama.cpp embeds exactly the string it is sent -- so a remote
# backend has to carry them itself, and getting this wrong is invisible: the
# vectors are still 1024 well-formed numbers, they simply answer a slightly
# different question than the stored ones do.
_KNOWN_PROMPTS: dict[str, dict[str, str]] = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "query": (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:"
        ),
        "document": "",
    },
}
_KNOWN_PROMPTS["Qwen/Qwen3-Embedding-4B"] = _KNOWN_PROMPTS["Qwen/Qwen3-Embedding-0.6B"]
_KNOWN_PROMPTS["Qwen/Qwen3-Embedding-8B"] = _KNOWN_PROMPTS["Qwen/Qwen3-Embedding-0.6B"]


@runtime_checkable
class Embedder(Protocol):
    """What the builder and the search server need from an embedding backend."""

    @property
    def model_name(self) -> str:
        """The name stamped on the vector collection, not the endpoint's id."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def on_device(self) -> Any: ...


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


class OpenAICompatibleEmbedder:
    """Embeds through an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Interchangeable with SentenceTransformerEmbedder for both callers, with two
    differences that are the whole reason it exists and the whole reason it is
    not the default:

    * It needs no torch and no resident model, so the search server -- which
      embeds exactly one short query per request -- stops paying for a 0.6B
      model it uses for a few milliseconds at a time.
    * It is an HTTP round trip per batch, so embedding a whole corpus through
      it is far slower than the builder's GPU. The builder stays on "local".

    ``model_name`` deliberately reports ``settings.EMBEDDING_MODEL`` rather than
    the id the endpoint was called with. The collection records which *vector
    space* it holds; whether today's vectors came from a local checkpoint or
    from a GGUF of the same model behind a server is a deployment detail, and
    making it part of the recorded identity would reject a database that is in
    fact perfectly readable.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_model: str,
        api_key: str,
        query_prefix: Optional[str],
        document_prefix: Optional[str],
        batch_size: int,
        timeout: float,
        max_retries: int,
    ):
        if not api_base:
            raise ValueError(
                "EMBEDDING_BACKEND is 'openai' but EMBEDDING_API_BASE is empty."
            )

        self._model_name = model_name
        self._url = api_base.rstrip("/") + "/embeddings"
        self._api_model = api_model or model_name
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._dimension: Optional[int] = None

        prompts = _KNOWN_PROMPTS.get(model_name)
        if prompts is None and (query_prefix is None or document_prefix is None):
            logger.warning(
                "No stored prompts are known for '%s' and the remote backend "
                "cannot read them from the model. Falling back to no prefix; "
                "set EMBEDDING_QUERY_PREFIX and EMBEDDING_DOCUMENT_PREFIX "
                "explicitly if the model expects one, or queries will search a "
                "different space than the stored vectors occupy.",
                model_name,
            )
        prompts = prompts or {}
        self._query_prefix = (
            query_prefix if query_prefix is not None else prompts.get("query", "")
        )
        self._document_prefix = (
            document_prefix
            if document_prefix is not None
            else prompts.get("document", "")
        )
        logger.info("Resolved embedding query prefix: %r", self._query_prefix)
        logger.info("Resolved embedding document prefix: %r", self._document_prefix)

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
        """Asks the endpoint, once, by embedding a probe string."""
        if self._dimension is None:
            self._dimension = len(self._post([""])[0])
        return self._dimension

    def _post(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._api_model, "input": texts}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = httpx.post(
                    self._url, json=payload, headers=headers, timeout=self._timeout
                )
                response.raise_for_status()
                body = response.json()
            except Exception as e:  # noqa: BLE001 - retried below, re-raised after
                last_error = e
                if attempt + 1 < self._max_retries:
                    delay = 2.0**attempt
                    logger.warning(
                        "Embedding request failed (%s); retrying in %.0fs "
                        "(attempt %d/%d).",
                        e, delay, attempt + 1, self._max_retries,
                    )
                    time.sleep(delay)
                continue

            # The response is not promised to come back in request order.
            data = sorted(body["data"], key=lambda item: item.get("index", 0))
            if len(data) != len(texts):
                raise RuntimeError(
                    f"Embedding endpoint returned {len(data)} vectors for "
                    f"{len(texts)} inputs."
                )
            return [item["embedding"] for item in data]

        raise RuntimeError(
            f"Embedding endpoint {self._url} failed after "
            f"{self._max_retries} attempts: {last_error}"
        ) from last_error

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [prefix + text for text in texts] if prefix else list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self._batch_size):
            vectors.extend(self._post(prefixed[start : start + self._batch_size]))

        # The application's collections use cosine, and the local backend
        # normalises. An endpoint is not obliged to, and an unnormalised vector
        # would not fail -- it would just rank badly -- so it is done here.
        return [_normalize(vector) for vector in vectors]

    @contextlib.contextmanager
    def on_device(self):
        """No-op: the model is somebody else's process."""
        yield

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), self._document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self._query_prefix)[0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def get_embedder() -> Optional[Embedder]:
    """
    Builds the embedder from the application settings.

    Returns None (after logging an actionable error) when the selected backend
    is unusable: sentence-transformers or torch not installed for "local", or a
    missing/unreachable endpoint for "openai".
    """
    backend = settings.EMBEDDING_BACKEND.strip().lower()

    if backend == "openai":
        logger.info(
            f"Embedding via {settings.EMBEDDING_API_BASE} as "
            f"'{settings.EMBEDDING_API_MODEL or settings.EMBEDDING_MODEL}' "
            f"(recorded as {settings.EMBEDDING_MODEL})"
        )
        try:
            return OpenAICompatibleEmbedder(
                model_name=settings.EMBEDDING_MODEL,
                api_base=settings.EMBEDDING_API_BASE,
                api_model=settings.EMBEDDING_API_MODEL,
                api_key=settings.EMBEDDING_API_KEY,
                query_prefix=settings.EMBEDDING_QUERY_PREFIX,
                document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
                batch_size=settings.EMBEDDING_API_BATCH_SIZE,
                timeout=settings.EMBEDDING_API_TIMEOUT,
                max_retries=settings.EMBEDDING_API_MAX_RETRIES,
            )
        except (ImportError, ValueError) as e:
            logger.error(f"Remote embedding backend is unusable: {e}")
            return None

    if backend != "local":
        logger.error(
            f"EMBEDDING_BACKEND is '{settings.EMBEDDING_BACKEND}'; "
            "expected 'local' or 'openai'."
        )
        return None

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


# HNSW graph parameters. Chroma's defaults (max_neighbors 16,
# ef_construction 100) are built for smaller collections and leave a
# half-million-vector index fragile enough that the *order* the vectors arrive
# in decides how good it is. Measured on this corpus -- 504,346 chunks, 50 real
# questions, recall@5 against an exact full scan:
#
#     built incrementally, Chroma defaults          89.2%
#     imported from an archive, Chroma defaults     60.4%  and  67.6%
#     imported from an archive, the values below    96.8%
#
# The two import runs are the same code and the same data; HNSW draws node
# levels at random, so builds differ by several points on their own. What they
# have in common is arriving grouped by manual, which is how an archive hands
# them over, and that costs roughly 25 points at the default settings. Raising
# the two parameters removes the dependence on order entirely -- the 96.8% run
# was inserted in that same worst-case order and still beat the incremental
# build. Japanese queries gain most (56.0% -> 97.0%): a sparse graph strands
# them in a local minimum more often than English ones.
#
# The cost is small: an import goes from 17:59 to 20:47, the index grows about
# 1%, and a query goes from 2.2 ms to 2.7 ms.
#
# These apply only when a collection is *created*. An existing database keeps
# the parameters it was built with; re-import the archive into a fresh one.
HNSW_MAX_NEIGHBORS = 32
HNSW_EF_CONSTRUCTION = 200


def collection_metadata(embedder: Embedder) -> dict[str, Any]:
    """Metadata stored on the Chroma collection when it is first created."""
    return {
        EMBEDDING_MODEL_METADATA_KEY: embedder.model_name,
        "embedding_dim": embedder.dimension,
        "hnsw:space": "cosine",
        "hnsw:M": HNSW_MAX_NEIGHBORS,
        "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
        "description": "Chunks from PDF manuals",
    }


def check_embedding_model(stored_model: Optional[str], expected_model: str) -> None:
    """
    Verifies that stored vectors came from the expected model.

    Vectors from different models are not comparable, so a mismatch has to stop
    the caller instead of silently returning nonsense results.
    """
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


def check_collection_model(collection: Any, expected_model: str) -> None:
    """Verifies a Chroma collection object was built with the expected model."""
    metadata = getattr(collection, "metadata", None) or {}
    check_embedding_model(metadata.get(EMBEDDING_MODEL_METADATA_KEY), expected_model)
