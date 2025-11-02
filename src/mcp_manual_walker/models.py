from datetime import datetime, UTC
import uuid
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    PrimaryKeyConstraint,
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
    page_count: Mapped[int] = Column(Integer, nullable=False)
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    bookmarks: Mapped[List["Bookmark"]] = relationship("Bookmark", back_populates="manual", cascade="all, delete-orphan")
    cache_entries: Mapped[List["Cache"]] = relationship("Cache", back_populates="manual", cascade="all, delete-orphan")

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

class Cache(Base):
    __tablename__ = "cache"

    manual_id: Mapped[str] = Column(String(36), ForeignKey("manuals.id"), nullable=False)
    page_num: Mapped[int] = Column(Integer, nullable=False)
    manual_hash: Mapped[str] = Column(String, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    last_accessed_at: Mapped[datetime] = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    manual: Mapped["Manual"] = relationship("Manual", back_populates="cache_entries")

    __table_args__ = (
        PrimaryKeyConstraint('manual_id', 'page_num'),
    )