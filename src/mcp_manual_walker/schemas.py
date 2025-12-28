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
        ...,
        description="The complete Markdown content for the requested bookmark section and its subsections.",
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


class SearchResult(BaseModel):
    manual_id: str = Field(..., description="The unique identifier of the manual.")
    query: str = Field(..., description="The search query used.")
    results: List[SearchResultItem] = Field(
        ..., description="A list of search results found in the manual."
    )
