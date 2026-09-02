from typing import List, Optional

from pydantic import BaseModel, Field


class ManualInfo(BaseModel):
    id: str = Field(..., description="The unique identifier for the manual.")
    file_name: str = Field(..., description="The filename of the manual PDF.")
    document_title: Optional[str] = Field(
        None, description="The title of the document as extracted from PDF metadata."
    )


class BookmarkNode(BaseModel):
    id: str = Field(..., description="The unique identifier for the bookmark.")
    title: str = Field(..., description="The title of the bookmark.")
    page: int = Field(..., description="The page number where the bookmark is located.")
    children: List["BookmarkNode"] = Field(
        ..., description="A list of nested child bookmarks."
    )


class ManualMetadata(ManualInfo):
    file_hash: str = Field(..., description="The SHA256 hash of the PDF file.")
    table_of_contents: List[BookmarkNode] = Field(
        ..., description="The hierarchical table of contents derived from bookmarks."
    )


class FigureRef(BaseModel):
    id: str = Field(..., description="The unique identifier of the figure.")
    page: int = Field(..., description="The page the figure appears on.")
    caption: Optional[str] = Field(
        None, description="The caption printed next to the figure, if any."
    )
    description: Optional[str] = Field(
        None,
        description="A textual description of the figure generated at build time.",
    )
    bookmark_id: Optional[str] = Field(
        None, description="The unique identifier of the bookmark the figure belongs to."
    )


class FigureInfo(FigureRef):
    manual_id: str = Field(..., description="The unique identifier of the manual.")
    labels: Optional[str] = Field(
        None, description="Comma-joined text labels drawn inside the figure."
    )
    width: Optional[int] = Field(None, description="The image width in pixels.")
    height: Optional[int] = Field(None, description="The image height in pixels.")
    mime_type: str = Field(..., description="The MIME type of the image bytes.")


class MarkdownContent(BaseModel):
    markdown_content: str = Field(
        ...,
        description="The complete Markdown content for the requested bookmark section and its subsections.",
    )
    figures: List[FigureRef] = Field(
        default_factory=list,
        description=(
            "The figures appearing in the returned section, in document order. "
            "Use `get_figure` with an id to fetch the image itself."
        ),
    )


class SearchResultItem(BaseModel):
    bookmarks: List[BookmarkNode] = Field(
        ..., description="The hierarchy of bookmarks leading to this result."
    )
    context: str = Field(
        ..., description="The full text content of the matching chunk."
    )
    manual_id: str = Field(..., description="The unique identifier of the manual.")
    bookmark_id: Optional[str] = Field(
        None, description="The unique identifier of the bookmark this chunk belongs to."
    )
    chunk_type: str = Field(
        "text",
        description='The kind of chunk that matched: "text", "table" or "figure".',
    )
    figure: Optional[FigureRef] = Field(
        None,
        description=(
            "The figure this chunk describes, set only for figure chunks. "
            "Use `get_figure` with its id to fetch the image itself."
        ),
    )


class SearchResult(BaseModel):
    manual_id: str = Field(..., description="The unique identifier of the manual.")
    query: str = Field(..., description="The search query used.")
    results: List[SearchResultItem] = Field(
        ..., description="A list of search results found in the manual."
    )
