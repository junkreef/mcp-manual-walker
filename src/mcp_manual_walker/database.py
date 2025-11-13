from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import settings
from .models import Base

engine = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def init_db():
    global engine
    engine = create_engine(f"sqlite:///{settings.DB_FILE_PATH}")
    SessionLocal.configure(bind=engine)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()