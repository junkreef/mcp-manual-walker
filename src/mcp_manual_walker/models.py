import uuid
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, declarative_base, relationship

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Manual(Base):
    __tablename__ = "manuals"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    file_name: Mapped[str] = Column(String, nullable=False)
    document_title: Mapped[Optional[str]] = Column(String)
    relative_path: Mapped[str] = Column(String, unique=True, nullable=False)
    file_hash: Mapped[str] = Column(String, nullable=False)
    page_count: Mapped[int] = Column(Integer, nullable=False)
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    # Set once the manual's chunks have reached ChromaDB. A row exists from the
    # moment the metadata pass syncs it, long before it is converted, so the
    # file hash alone cannot tell a finished manual from one an interrupted
    # build never got to -- see `builder.build`.
    converted_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True), nullable=True
    )

    bookmarks: Mapped[List["Bookmark"]] = relationship(
        "Bookmark", back_populates="manual", cascade="all, delete-orphan"
    )
    figures: Mapped[List["Figure"]] = relationship(
        "Figure", back_populates="manual", cascade="all, delete-orphan"
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


class VectorStoreMeta(Base):
    """Facts about the vector store that the vector store cannot hold itself.

    Chroma records the embedding model in its collection metadata, which is
    what the server checks before serving: vectors from two models are not
    comparable, and a mismatch has to stop the caller rather than quietly
    return nonsense. Qdrant has no equivalent -- a collection carries its
    vector parameters and nothing else -- so for backends like it the name is
    kept here instead, beside the manuals whose chunks those vectors are.

    The cost of that choice is that the relational database and the vector
    database now have to belong to each other. They already did in practice:
    a chunk id means nothing without the bookmarks table, and a figure id
    means nothing without the figures table.
    """

    __tablename__ = "vector_store_meta"

    key: Mapped[str] = Column(String, primary_key=True)
    value: Mapped[Optional[str]] = Column(Text, nullable=True)
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class Figure(Base):
    """A picture detected by Docling, stored as a PNG blob.

    The image bytes live in SQLite so that the database file and the Chroma
    directory are the only artifacts that have to travel between machines;
    chunk metadata in Chroma only carries the figure id.
    """

    __tablename__ = "figures"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    manual_id: Mapped[str] = Column(
        String(36), ForeignKey("manuals.id"), nullable=False
    )
    # Plain column, deliberately without a foreign key: bookmarks are re-created
    # on every manual update, and Chroma stores the same plain id.
    bookmark_id: Mapped[Optional[str]] = Column(String(36), nullable=True)
    picture_index: Mapped[int] = Column(Integer, nullable=False)
    page: Mapped[int] = Column(Integer, nullable=False)
    # Bounding box in PDF points, bottom-left origin (Docling prov bbox).
    bbox_l: Mapped[float] = Column(Float, nullable=False)
    bbox_b: Mapped[float] = Column(Float, nullable=False)
    bbox_r: Mapped[float] = Column(Float, nullable=False)
    bbox_t: Mapped[float] = Column(Float, nullable=False)
    caption: Mapped[Optional[str]] = Column(Text, nullable=True)
    # Comma-joined text labels drawn inside the picture.
    labels: Mapped[Optional[str]] = Column(Text, nullable=True)
    # Filled by a later phase (vision model description).
    description: Mapped[Optional[str]] = Column(Text, nullable=True)
    mime_type: Mapped[str] = Column(String, default="image/png")
    width: Mapped[Optional[int]] = Column(Integer)
    height: Mapped[Optional[int]] = Column(Integer)
    image: Mapped[bytes] = Column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    manual: Mapped["Manual"] = relationship("Manual", back_populates="figures")
