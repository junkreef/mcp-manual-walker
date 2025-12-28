import uuid
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, declarative_base, relationship

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
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    bookmarks: Mapped[List["Bookmark"]] = relationship(
        "Bookmark", back_populates="manual", cascade="all, delete-orphan"
    )


class Bookmark(Base):
    __tablename__ = "bookmarks"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    manual_id: Mapped[str] = Column(
        String(36), ForeignKey("manuals.id"), nullable=False
    )
    ordering: Mapped[int] = Column(Integer, nullable=False)
    title: Mapped[str] = Column(String, nullable=False)
    level: Mapped[int] = Column(Integer, nullable=False)
    page_num: Mapped[int] = Column(Integer, nullable=False)
    page_top: Mapped[Optional[float]] = Column(Float, nullable=True)
    parent_id: Mapped[Optional[str]] = Column(String(36), ForeignKey("bookmarks.id"))

    manual: Mapped["Manual"] = relationship("Manual", back_populates="bookmarks")
    parent: Mapped[Optional["Bookmark"]] = relationship(
        "Bookmark", remote_side=[id], back_populates="children"
    )
    children: Mapped[List["Bookmark"]] = relationship(
        "Bookmark", back_populates="parent", cascade="all, delete-orphan"
    )
