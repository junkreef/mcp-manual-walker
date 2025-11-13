from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from mcp_manual_walker import config
from mcp_manual_walker.database import SessionLocal, init_db
from mcp_manual_walker.models import Manual
from mcp_manual_walker.sync import sync_database


@pytest.fixture(scope="function")
def db_session():
    """Provides a database session for a test."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def test_sync_environment(tmp_path: Path, monkeypatch, dummy_pdf_factory):
    """
    Sets up a temporary, isolated environment for testing sync_database.
    - Creates temporary directories for PDFs, database, and cache.
    - Mocks the application settings to use these temporary paths.
    - Provides a factory for creating dummy PDFs within the test environment.
    - Yields the paths to the directories for manipulation during tests.
    """
    # 1. Create temporary directories
    pdf_dir = tmp_path / "pdfs"
    db_dir = tmp_path / "db"
    cache_dir = tmp_path / "cache"
    pdf_dir.mkdir()
    db_dir.mkdir()
    cache_dir.mkdir()

    # 2. Configure app settings to use temporary paths
    monkeypatch.setattr(config.settings, "PDF_ROOT_DIR", pdf_dir)
    monkeypatch.setattr(config.settings, "DB_FILE_PATH", db_dir / "test.db")
    monkeypatch.setattr(config.settings, "CACHE_DIR", cache_dir)

    # 3. Initialize the database in the temporary location
    init_db()

    # 4. Yield the paths for the test function to use
    yield pdf_dir, cache_dir, dummy_pdf_factory


def test_sync_database_lifecycle(test_sync_environment, db_session: Session):
    """
    Tests the full lifecycle of the sync_database function:
    1. Initial sync: Adds new files.
    2. Second sync: Updates a modified file and adds a new one.
    3. Third sync: Deletes a removed file and its cache.
    """
    pdf_dir, cache_dir, dummy_pdf_factory = test_sync_environment

    # --- 1. Initial Sync (Addition) ---
    print("\n--- Running Initial Sync ---")
    pdf_a_path = pdf_dir / "manual_A.pdf"
    pdf_b_path = pdf_dir / "manual_B.pdf"

    dummy_pdf_factory(pdf_a_path, {1: "Content A v1"}, metadata={"/Title": "Manual A"})
    dummy_pdf_factory(pdf_b_path, {1: "Content B"}, metadata={"/Title": "Manual B"})

    sync_database()

    manuals = db_session.query(Manual).order_by(Manual.file_name).all()
    assert len(manuals) == 2
    manual_a, manual_b = manuals
    assert manual_a.file_name == "manual_A.pdf"
    assert manual_b.file_name == "manual_B.pdf"
    hash_a_v1 = manual_a.file_hash

    # Create a dummy cache directory for manual_b to test cleanup later
    manual_b_cache_dir = cache_dir / manual_b.id
    manual_b_cache_dir.mkdir()
    (manual_b_cache_dir / "dummy_cache_file.md").touch()
    assert manual_b_cache_dir.exists()

    # --- 2. Second Sync (Update and Addition) ---
    # Update manual_A.pdf
    dummy_pdf_factory(
        pdf_a_path, {1: "Content A v2"}, metadata={"/Title": "Manual A Updated"}
    )
    # Add manual_C.pdf
    dummy_pdf_factory(
        pdf_dir / "manual_C.pdf", {1: "Content C"}, metadata={"/Title": "Manual C"}
    )

    sync_database()

    manuals = db_session.query(Manual).order_by(Manual.file_name).all()
    assert len(manuals) == 3
    manual_a, manual_b, manual_c = manuals
    assert manual_a.file_name == "manual_A.pdf"
    assert manual_a.file_hash != hash_a_v1  # Hash should be updated
    assert manual_b.file_name == "manual_B.pdf"
    assert manual_c.file_name == "manual_C.pdf"

    # --- 3. Third Sync (Deletion) ---
    pdf_b_path.unlink()  # Remove manual_B.pdf from filesystem

    sync_database()

    manuals = db_session.query(Manual).order_by(Manual.file_name).all()
    assert len(manuals) == 2
    assert manuals[0].file_name == "manual_A.pdf"
    assert manuals[1].file_name == "manual_C.pdf"

    # Check that the cache directory for the deleted manual is also gone
    assert not manual_b_cache_dir.exists()
