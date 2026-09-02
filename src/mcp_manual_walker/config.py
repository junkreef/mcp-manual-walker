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
    # None means: use the prompt the model ships under the names "query" /
    # "document" (Qwen3-Embedding: an instruction on queries, nothing on
    # documents). An explicit string (possibly empty) overrides it.
    EMBEDDING_QUERY_PREFIX: Optional[str] = None
    EMBEDDING_DOCUMENT_PREFIX: Optional[str] = None
    # Tokens per input; Qwen3 supports 32k but VRAM/RAM grows with the window
    EMBEDDING_MAX_SEQ_LENGTH: int = 4096
    EMBEDDING_BATCH_SIZE: int = 32


settings = Settings()
