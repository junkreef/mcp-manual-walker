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


class MarkdownContent(BaseModel):
    markdown_content: str = Field(
        ..., description="The converted Markdown content for the requested page range."
    )
    bookmark_total_pages: int = Field(
        ..., description="The total number of pages within this bookmark section."
    )
    page_offset: int = Field(
        ...,
        description="The starting page offset of the returned content, " \
        "0-indexed from the beginning of the bookmark section.",
    )
    page_limit: int = Field(
        ..., description="The number of pages returned in this response."
    )
    next_page_offset: Optional[int] = Field(
        None,
        description="The offset to use in the next request to get the following pages. " \
        "If null, this is the last page of the section.",
    )
