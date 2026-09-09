import multiprocessing
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    PDF_ROOT_DIR: Path = Path("./data/pdfs")
    DB_FILE_PATH: Path = Path("./data/mcp_manual_walker.db")
    # Which vector backend holds the chunks: "chroma" or "qdrant".
    #
    # Chroma is embedded and needs no server, which is right up to the point
    # where its hnswlib arena stops fitting in RAM -- roughly a million
    # 1024-dimensional chunks. Qdrant memory-maps the same data and needs a
    # server process. Measured on 684,398 chunks: Chroma about 2.9 GB the
    # process must have, Qdrant about 350 MB it must have plus 2.7 GB the
    # kernel may reclaim. Under a 512 MB container cap Qdrant kept answering,
    # which says the process does not need the memory rather than that the
    # machine does not -- the host still had page cache to give it.
    VECTOR_BACKEND: str = "chroma"
    CHROMADB_PATH: Path = Path("./data/db/chroma_db")

    # Qdrant. Only consulted when VECTOR_BACKEND is "qdrant".
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_GRPC_PORT: int = 6334
    # gRPC rather than HTTP: a build uploads hundreds of thousands of 4 kB
    # vectors, and JSON-encoding them is the expensive part.
    QDRANT_PREFER_GRPC: bool = True
    QDRANT_COLLECTION: str = "manual_chunks"
    QDRANT_TIMEOUT: float = 60.0
    # Candidates held during a graph traversal. Higher means better recall and
    # a slower query; 128 measured 96.2% recall@5 against an exact scan of the
    # same collection, corpus-wide.
    QDRANT_HNSW_EF: int = 128
    # The three memory knobs, all off because on this corpus all three made
    # matters worse. Qdrant already memory-maps vectors it calls "in RAM", so
    # moving them "to disk" buys nothing, and int8 with always_ram *adds* about
    # 700 MB of unreclaimable memory while taking filtered search from 2.6 ms
    # to 20.4 ms. They are here because a corpus several times this one may
    # answer differently.
    QDRANT_ON_DISK_VECTORS: bool = False
    QDRANT_ON_DISK_PAYLOAD: bool = False
    # "none" | "int8" | "binary"
    QDRANT_QUANTIZATION: str = "none"
    # Shortlist multiplier used to rescore a quantized search. Ignored when
    # QDRANT_QUANTIZATION is "none".
    QDRANT_OVERSAMPLING: float = 2.0
    MARKDOWN_OUTPUT_DIR: Path = Path("./data/markdown")
    # Append-only JSONL log of per-file build progress, truncated at the start
    # of every build and read by `db_manager watch`. Purely observational: the
    # build never reads it back.
    BUILD_PROGRESS_FILE: Path = Path("./data/build_progress.jsonl")
    LOG_LEVEL: str = "INFO"
    MAX_PAGES_PER_REQUEST: int = 20

    # Chunking
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    CHUNK_OVERLAP_SEARCH_MARGIN: int = 100

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Docling Configuration
    # DOCLING_NUM_THREADS is the TOTAL CPU thread budget shared by all Docling
    # worker processes; each worker gets DOCLING_NUM_THREADS // DOCLING_WORKERS.
    DOCLING_NUM_THREADS: int = multiprocessing.cpu_count()
    DOCLING_OCR_BATCH_SIZE: int = 16
    DOCLING_LAYOUT_BATCH_SIZE: int = 16
    DOCLING_TABLE_BATCH_SIZE: int = 16
    # Pages allowed to sit in each pipeline stage's input queue. Docling defaults
    # to 100 across six stages, so a long document keeps ~500 rendered page images
    # resident at once and peak RSS is set by this, not by the page count. Measured
    # on a 600-page manual: 100 -> 8.2 GB peak, 16 -> 5.0 GB, with no change to the
    # output and none to the wall time.
    DOCLING_QUEUE_MAX_SIZE: int = 16
    # Pages per conversion unit. A document longer than this is converted as
    # several page ranges, in parallel across the worker pool, and merged in
    # the parent. This is what makes a worker's peak memory a function of a
    # number you choose rather than of the longest document in the corpus, and
    # it stops one 2900-page manual from occupying a single worker while the
    # rest of the GPU idles. Measured on a 1246-page manual: 403 s / 5.93 GB
    # converted whole, 286 s / 3.94 GB peak worker as 250-page parts across 3
    # workers, with byte-identical output. 0 disables splitting.
    DOCLING_SPLIT_PAGES: int = 250
    # How many things may use the GPU at once. Docling workers and the
    # builder's own embedding model share one device and neither yields to the
    # other, so on a 23 GB L4 three converting workers (17 GB) plus an
    # embedding batch (5 GB) is 22 GB and something fails -- observed three
    # different ways in one afternoon: the embedder unable to allocate 192 MB,
    # RapidOCR's arena unable to allocate 142 MB, and a document lost outright.
    # A slot is held for the length of one conversion or one embedding call, so
    # embedding a finished document simply costs one converting worker until it
    # is done. 0 disables the limit and lets everything race as before.
    DOCLING_GPU_SLOTS: int = 3
    # Render scale for the figure crops persisted to SQLite (1.0 = 72 dpi,
    # 2.0 = 144 dpi). Higher values give sharper PNGs and a bigger database.
    DOCLING_IMAGES_SCALE: float = 2.0

    # Worker pipeline configuration
    # pypdf hash/outline extraction processes; <=1 means run inline
    METADATA_WORKERS: int = max(1, multiprocessing.cpu_count() // 2)
    # Number of Docling converter processes (each loads its own models into VRAM)
    DOCLING_WORKERS: int = 1
    # "auto" | "cpu" | "cuda" | "cuda:N" | "mps", passed to AcceleratorOptions
    DOCLING_DEVICE: str = "auto"
    # RapidOCR inference backend passed to RapidOcrOptions(backend=...):
    # "onnxruntime" (default; "japan"/"chinese"/"en" models ship inside the
    # rapidocr wheel, works offline, GPU via onnxruntime-gpu) or "torch"
    # (always downloads .pth checkpoints from modelscope.cn on first use).
    DOCLING_OCR_BACKEND: str = "onnxruntime"
    # RapidOCR language token for the recognition model. "japan", "chinese"
    # and "en" resolve to the bundled PP-OCRv6 ONNX models (no download);
    # other languages (e.g. "korean") are fetched from modelscope.cn.
    DOCLING_OCR_LANG: str = "japan"
    # Embedding model (shared by the builder and the search server; must match the DB)
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    # Where the vectors are produced.
    #   "local"  load the model with SentenceTransformers in this process
    #   "openai" call an OpenAI-compatible /v1/embeddings endpoint
    # The search server embeds one query per request, which is all a remote
    # endpoint has to be good at, and in exchange it needs neither torch nor a
    # resident copy of the model. The builder embeds the whole corpus and wants
    # the GPU it already has, so it stays on "local".
    EMBEDDING_BACKEND: str = "local"
    # Base URL of the OpenAI-compatible API, e.g. "http://localhost:11434/v1"
    # (Ollama), "http://localhost:8000/v1" (vLLM), or a Lemonade Server.
    EMBEDDING_API_BASE: str = ""
    # Sent as "Authorization: Bearer ******" when set.
    EMBEDDING_API_KEY: str = ""
    # The id the *endpoint* knows the model by, which is a deployment detail and
    # is often not EMBEDDING_MODEL: a Lemonade Server serving Qwen3-Embedding-0.6B
    # as a GGUF calls it "Qwen3-Embedding-0.6B-GGUF". Empty means: use
    # EMBEDDING_MODEL. EMBEDDING_MODEL remains the name stamped on the vector
    # collection, because that is what identifies the vector space.
    EMBEDDING_API_MODEL: str = ""
    EMBEDDING_API_TIMEOUT: float = 120.0
    # Texts per request. Kept modest because a request carries whole chunks.
    EMBEDDING_API_BATCH_SIZE: int = 16
    EMBEDDING_API_MAX_RETRIES: int = 3
    # "auto" | "cpu" | "cuda" ... for SentenceTransformers
    EMBEDDING_DEVICE: str = "auto"
    # Torch dtype the weights are loaded under. "auto" takes the dtype from the
    # checkpoint (bfloat16 for Qwen3-Embedding); any torch dtype name
    # ("float32", "bfloat16", "float16") forces that instead. This is passed
    # explicitly rather than left to the library default, which has already
    # flipped once between transformers 4.x (float32) and 5.x (auto).
    # Prefer "float32" on a CPU-only search server: bfloat16 has no fast CPU
    # kernels, and query embedding measured ~2x slower than float32 on one.
    EMBEDDING_DTYPE: str = "auto"
    # None means: use the prompt the model ships under the names "query" /
    # "document" (Qwen3-Embedding: an instruction on queries, nothing on
    # documents). An explicit string (possibly empty) overrides it.
    EMBEDDING_QUERY_PREFIX: Optional[str] = None
    EMBEDDING_DOCUMENT_PREFIX: Optional[str] = None
    # Tokens per input; Qwen3 supports 32k but VRAM/RAM grows with the window
    EMBEDDING_MAX_SEQ_LENGTH: int = 4096
    EMBEDDING_BATCH_SIZE: int = 32
    # Upper bound on the padded tokens in one encode call, i.e.
    # len(batch) x the longest text in it. A batch is padded to its longest
    # member, so budgeting rows alone prices every batch at its worst one: on
    # 562 real chunks, one 3836-token figure caption among texts averaging 298
    # tokens took the peak from 6.7 GB to 18.6 GB of VRAM. Measured at roughly
    # 0.15 MB of VRAM per padded token on an L4 with Qwen3-Embedding-0.6B, so
    # 24576 tokens is about 3.7 GB of activations, leaving the GPU to the
    # Docling workers. 0 falls back to plain row batching.
    EMBEDDING_TOKEN_BUDGET: int = 24576

    # Figure descriptions (optional)
    # Ask a local OpenAI-compatible vision model to describe every detected
    # figure. Empty (the default) disables the feature entirely.
    # Ollama: http://localhost:11434/v1/chat/completions
    # llama.cpp server: http://localhost:8080/v1/chat/completions
    PICTURE_DESCRIPTION_URL: str = ""
    # Sent as "model" in the request payload when set. Ollama requires it;
    # llama.cpp's server ignores it (the model is fixed at server startup).
    PICTURE_DESCRIPTION_MODEL: str = ""
    # Sent as the "Authorization: Bearer ..." header when set.
    PICTURE_DESCRIPTION_API_KEY: str = ""
    # Prompt sent alongside each figure crop.
    PICTURE_DESCRIPTION_PROMPT: str = (
        "これは技術マニュアルに掲載された図です。図が示している機器・部品・接続・"
        "操作手順を、検索用の要約として日本語で簡潔に説明してください。"
        "図中に書かれている文字はそのまま含めてください。"
    )
    # Upper bound on the generated description length.
    PICTURE_DESCRIPTION_MAX_TOKENS: int = 300
    # HTTP timeout per request to the vision API, in seconds.
    PICTURE_DESCRIPTION_TIMEOUT: float = 120.0
    # Parallel requests in flight per Docling worker.
    PICTURE_DESCRIPTION_CONCURRENCY: int = 1
    # Pictures smaller than this fraction of the page area are skipped.
    PICTURE_DESCRIPTION_AREA_THRESHOLD: float = 0.02


settings = Settings()
