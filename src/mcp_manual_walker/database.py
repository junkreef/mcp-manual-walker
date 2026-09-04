import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from .config import settings
from .models import Base

logger = logging.getLogger(__name__)

engine = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

# Columns added to a table after the first release, as
# {table: {column: DDL type}}. `create_all` only ever creates missing tables,
# so a database built by an earlier version keeps its old table definition and
# every query naming the new column fails. There is no migration tool in this
# project; this covers the one case it needs to.
_ADDED_COLUMNS = {
    "manuals": {"converted_at": "DATETIME"},
}


def _add_missing_columns(bind):
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    with bind.begin() as connection:
        for table, columns in _ADDED_COLUMNS.items():
            if table not in existing_tables:
                # create_all just made it, with every column.
                continue
            present = {col["name"] for col in inspector.get_columns(table)}
            for column, ddl_type in columns.items():
                if column in present:
                    continue
                logger.info(f"Adding column {table}.{column} to the database.")
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                )


def init_db():
    global engine
    engine = create_engine(f"sqlite:///{settings.DB_FILE_PATH}")
    SessionLocal.configure(bind=engine)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
