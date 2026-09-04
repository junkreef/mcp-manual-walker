"""Resuming a build that did not finish.

A manual's row is written by the metadata pass, long before the document is
converted, so the file hash alone cannot distinguish a finished manual from
one an interrupted build never reached. Without the `converted_at` marker a
re-run reports every file as "unchanged" and stores nothing.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from mcp_manual_walker import database
from mcp_manual_walker.models import Base, Manual


def make_session(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def test_a_fresh_manual_is_not_marked_converted(tmp_path):
    session, _ = make_session(tmp_path / "t.db")
    manual = Manual(
        id="m1",
        file_name="a.pdf",
        relative_path="a.pdf",
        file_hash="h",
        page_count=5,
    )
    session.add(manual)
    session.commit()
    assert session.get(Manual, "m1").converted_at is None


def test_the_marker_round_trips(tmp_path):
    session, _ = make_session(tmp_path / "t.db")
    now = datetime.now(UTC)
    session.add(
        Manual(
            id="m1",
            file_name="a.pdf",
            relative_path="a.pdf",
            file_hash="h",
            page_count=5,
            converted_at=now,
        )
    )
    session.commit()
    assert session.get(Manual, "m1").converted_at is not None


# -- the migration -----------------------------------------------------------


def test_init_db_adds_the_column_to_a_database_that_predates_it(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "old.db"
    # A database as an earlier version of this project would have left it:
    # the table exists, so create_all will not touch it.
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE manuals ("
                "id VARCHAR(36) PRIMARY KEY, file_name VARCHAR NOT NULL, "
                "document_title VARCHAR, relative_path VARCHAR NOT NULL UNIQUE, "
                "file_hash VARCHAR NOT NULL, page_count INTEGER NOT NULL, "
                "updated_at DATETIME)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO manuals (id, file_name, relative_path, file_hash,"
                " page_count) VALUES ('m1', 'a.pdf', 'a.pdf', 'h', 5)"
            )
        )
    engine.dispose()

    monkeypatch.setattr(database.settings, "DB_FILE_PATH", db_path)
    database.init_db()

    columns = {c["name"] for c in inspect(database.engine).get_columns("manuals")}
    assert "converted_at" in columns

    session = database.SessionLocal()
    try:
        # The existing row survives, and reads as not yet converted -- which
        # is the safe answer: it will be converted again rather than skipped.
        manual = session.get(Manual, "m1")
        assert manual is not None
        assert manual.converted_at is None
    finally:
        session.close()


def test_init_db_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "new.db"
    monkeypatch.setattr(database.settings, "DB_FILE_PATH", db_path)
    database.init_db()
    database.init_db()
    columns = [c["name"] for c in inspect(database.engine).get_columns("manuals")]
    assert columns.count("converted_at") == 1


@pytest.mark.parametrize("table", ["manuals"])
def test_every_declared_migration_names_a_real_column(table, tmp_path, monkeypatch):
    # A typo in _ADDED_COLUMNS would add a column the model never reads, and
    # nothing else would notice.
    monkeypatch.setattr(database.settings, "DB_FILE_PATH", tmp_path / "m.db")
    database.init_db()
    model_columns = {c.name for c in Base.metadata.tables[table].columns}
    for column in database._ADDED_COLUMNS[table]:
        assert column in model_columns
