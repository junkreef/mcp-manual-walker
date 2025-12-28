from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    PDF_ROOT_DIR: Path = Path("./data/pdfs")
    DB_FILE_PATH: Path = Path("./data/mcp_manual_walker.db")
    CHROMADB_PATH: Path = Path("./data/db/chroma_db")
    MARKDOWN_OUTPUT_DIR: Path = Path("./data/markdown")
    LOG_LEVEL: str = "INFO"
    MAX_PAGES_PER_REQUEST: int = 20

    HOST: str = "127.0.0.1"
    PORT: int = 8000


settings = Settings()
