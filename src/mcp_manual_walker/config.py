from pathlib import Path
import multiprocessing

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
    DOCLING_NUM_THREADS: int = multiprocessing.cpu_count()
    DOCLING_OCR_BATCH_SIZE: int = 16
    DOCLING_LAYOUT_BATCH_SIZE: int = 16
    DOCLING_TABLE_BATCH_SIZE: int = 16


settings = Settings()
