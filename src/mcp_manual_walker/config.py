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
    CHROMADB_PATH: Path = Path("./data/db/chroma_db")
    MARKDOWN_OUTPUT_DIR: Path = Path("./data/markdown")
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
