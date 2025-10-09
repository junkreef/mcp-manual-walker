import datetime
import uuid
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, Mapped

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Manual(Base):
    __tablename__ = "manuals"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    file_name: Mapped[str] = Column(String, unique=True, nullable=False)
    document_title: Mapped[Optional[str]] = Column(String)
    relative_path: Mapped[str] = Column(String, nullable=False)
    file_hash: Mapped[str] = Column(String, nullable=False)
    updated_at: Mapped[datetime.datetime] = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    bookmarks: Mapped[List["Bookmark"]] = relationship("Bookmark", back_populates="manual", cascade="all, delete-orphan")

class Bookmark(Base):
    __tablename__ = "bookmarks"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    manual_id: Mapped[str] = Column(String(36), ForeignKey("manuals.id"), nullable=False)
    ordering: Mapped[int] = Column(Integer, nullable=False)
    title: Mapped[str] = Column(String, nullable=False)
    level: Mapped[int] = Column(Integer, nullable=False)
    page_num: Mapped[int] = Column(Integer, nullable=False)
    parent_id: Mapped[Optional[str]] = Column(String(36), ForeignKey("bookmarks.id"))

    manual: Mapped["Manual"] = relationship("Manual", back_populates="bookmarks")
    parent: Mapped[Optional["Bookmark"]] = relationship("Bookmark", remote_side=[id], back_populates="children")
    children: Mapped[List["Bookmark"]] = relationship("Bookmark", back_populates="parent", cascade="all, delete-orphan")
    cache_entry: Mapped[Optional["Cache"]] = relationship("Cache", back_populates="bookmark", uselist=False, cascade="all, delete-orphan")


class Cache(Base):
    __tablename__ = "cache"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    bookmark_id: Mapped[str] = Column(String(36), ForeignKey("bookmarks.id"), nullable=False)
    manual_hash: Mapped[str] = Column(String, nullable=False)
    markdown_file_path: Mapped[str] = Column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = Column(DateTime, default=datetime.datetime.utcnow)

    bookmark: Mapped["Bookmark"] = relationship("Bookmark", back_populates="cache_entry")