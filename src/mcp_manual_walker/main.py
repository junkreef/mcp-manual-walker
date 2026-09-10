import json
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, List, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image
from pydantic import Field

from . import service
from .config import settings
from .schemas import (
    DirectoryEntry,
    ManualMetadata,
    MarkdownContent,
    SearchResult,
    SearchResultItem,
)
from .service import ServiceError
from .service import app_state as app_state  # re-exported: the tests reach for it here

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def _require_store():
    """Returns the vector store, or raises a ToolError explaining why it is gone."""
    with _as_tool_error():
        return service.require_store()


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Server startup event handler."""
    service.ensure_initialized()
    yield


app = FastMCP(lifespan=lifespan)


@contextmanager
def _as_tool_error():
    """Renders a service failure the way the MCP protocol expects.

    The service layer reports what went wrong and how, because the REST API has
    to turn that into a status code. An agent gets neither: every failure is a
    `ToolError` carrying the same sentence a human would have read.
    """
    try:
        yield
    except ServiceError as e:
        raise ToolError(str(e)) from e


@app.tool(
    name="list_manuals",
    description="""Browses the manual library one folder at a time, the way `ls` does.
    This tool is the primary entry point for discovering content. Called without
    arguments it returns the entries at the root of the library; called with the
    `folder` of a directory it returns what sits directly inside that directory.
    Nothing deeper is returned, so a large library can be explored without pulling
    hundreds of entries into the context at once.

    Every entry has a `type`:
    * `"directory"` — a folder. Pass its `path` back as `folder` to look inside.
      Its `manual_count` tells how many manuals it holds at any depth.
    * `"manual"` — a PDF. Its `id` is the `manual_id` the other tools need.

    Workflow Example:
    1. Call `list_manuals()` to see the top-level folders and manuals.
    2. Call `list_manuals(folder="Db2 for zOS")` to descend, repeating until the
       entries of type `"manual"` appear.
    3. Use the `id` of the manual you want to call `get_manual_metadata()` and
       retrieve its table of contents.""",
    tags={"manual", "discovery"},
    annotations={"readOnlyHint": True},
)
def list_manuals(
    folder: Annotated[
        str,
        Field(
            description="""The directory to list, relative to the library root and
            separated by `/` (for example `Db2 for zOS/v13.1`). Obtained from the
            `path` of a `"directory"` entry of a previous call. Leave it empty to
            list the root of the library."""
        ),
    ] = "",
) -> List[DirectoryEntry]:
    """Returns the manuals and subfolders directly inside the given folder."""
    with _as_tool_error():
        return service.list_folder(folder)


@app.tool(
    name="get_manual_metadata",
    description="""Retrieves detailed metadata and a hierarchical table of contents for
    a specific manual. Use this tool after you have identified a manual of interest
    using `list_manuals()`. It provides the full structure of the manual's bookmarks,
    which is essential for navigating its content. Each bookmark in the table of
    contents has its own unique ID, which is required by the `get_markdown_content`
    tool to fetch the actual content of that section.

    Workflow Example:
    1. Get a `manual_id` from the output of `list_manuals()`.
    2. Call `get_manual_metadata(manual_id=...)` to get the manual's structure.
    3. Browse the `table_of_contents` to find the specific section you need.
    4. Use the `id` of the desired bookmark to call `get_markdown_content()`.""",
    tags={"manual", "metadata", "toc"},
    annotations={"readOnlyHint": True},
)
def get_manual_metadata(
    manual_id: Annotated[
        str,
        Field(
            description="""The unique ID of the manual, 
            obtained from the `list_manuals` tool."""
        ),
    ],
) -> ManualMetadata:
    """Returns metadata and a hierarchical table of contents for a specified manual."""
    with _as_tool_error():
        return service.manual_metadata(manual_id)


@app.tool(
    name="get_markdown_content",
    description="""Fetches the Markdown content for a specific bookmark (section) within
      a manual using the Vector DB. This returns the pre-processed text chunks associated
      with the bookmark and its sub-sections.

    Figures (diagrams, drawings, screenshots) appear in the Markdown as a
    `[Figure: <figure_id> (page N)]` marker followed by the figure's caption,
    labels and description, and are also listed in the `figures` field in
    document order. Pass a figure id to `get_figure` to obtain the image itself.

    Workflow Example:
    1. Get a `bookmark_id` from the `table_of_contents` provided by
      `get_manual_metadata()`.
    2. Call `get_markdown_content(bookmark_id=...)` to get the content.
    3. Call `get_figure(figure_id=...)` for any figure you need to look at.""",
    tags={"manual", "content", "markdown"},
    annotations={"readOnlyHint": True},
)
def get_markdown_content(
    bookmark_id: Annotated[
        str,
        Field(
            description="""The unique ID of the bookmark, 
            obtained from `get_manual_metadata`."""
        ),
    ],
) -> MarkdownContent:
    """Returns the Markdown content for a specific bookmark from the Vector DB."""
    with _as_tool_error():
        return service.markdown_content(bookmark_id)


@app.tool(
    name="search_manual",
    description="""Searches for a query string within a specific manual using semantic search.
    Returns the top matching chunks.

    Optionally, a `bookmark_id` can be provided to restrict the search to a specific
    section of the manual (including subsections).

    Every result reports its `chunk_type` ("text", "table" or "figure"). A hit
    with `chunk_type` "figure" also carries a `figure` object whose `id` can be
    passed to `get_figure` to retrieve the image itself; its `context` is the
    figure's caption, labels and description.

    Workflow Example:
    1. Call `search_manual(manual_id=..., query="...")` to find occurrences.
    2. Call `get_figure(figure_id=...)` for a hit whose `chunk_type` is "figure".
    """,
    tags={"manual", "search"},
    annotations={"readOnlyHint": True},
)
def search_manual(
    manual_id: Annotated[
        str,
        Field(description="The unique ID of the manual to search."),
    ],
    query: Annotated[
        str,
        Field(description="The text to search for."),
    ],
    bookmark_id: Annotated[
        Optional[str],
        Field(
            description="Optional bookmark ID to restrict search to a specific section."
        ),
    ] = None,
) -> SearchResult:
    """Searches for text in a manual and returns matches with context and hierarchy."""
    with _as_tool_error():
        hits = service.search(query, manual_id=manual_id, bookmark_id=bookmark_id)

    # A hit carries more than the tool has ever returned -- its rank, its score
    # and the manual it came from -- because the REST client showing a result
    # list needs those. The tool response is narrowed back to what its schema
    # promises rather than quietly growing.
    return SearchResult(
        manual_id=manual_id,
        query=query,
        results=[SearchResultItem(**hit.model_dump()) for hit in hits],
    )


@app.tool(
    name="get_figure",
    description="""Returns the image of a figure (diagram, drawing, screenshot)
    stored from a manual, together with its metadata.
    Figure ids come from `search_manual` results whose `chunk_type` is "figure"
    (field `figure.id`) and from the `figures` list of `get_markdown_content`.
    The response contains the PNG image and a JSON text block with the figure's
    manual_id, bookmark_id, page, caption, labels, description and size.""",
    tags={"manual", "figure"},
    annotations={"readOnlyHint": True},
    # Required: fastmcp would otherwise try to serialize the Image object as
    # structured content, which fails.
    output_schema=None,
)
def get_figure(
    figure_id: Annotated[str, Field(description="The unique ID of the figure.")],
):
    """Returns the PNG image of a figure plus its metadata as JSON."""
    with _as_tool_error():
        image, info = service.get_figure(figure_id)

    return [
        Image(data=image, format="png"),
        json.dumps(info.model_dump()),
    ]


def serve() -> None:
    """Serves MCP on PORT and, unless disabled, the REST API on REST_PORT.

    The REST API runs in a thread of its own rather than as a second task on
    this loop. Both halves are uvicorn servers, and uvicorn installs its own
    SIGINT/SIGTERM handlers whenever it is started on the main thread: the
    second one to start would replace the first one's, and Ctrl-C would stop
    one server while the process went on running the other. Off the main
    thread uvicorn installs nothing, so the signal keeps reaching the MCP
    server, and this shuts the REST half down after it returns.
    """
    # Before the REST thread, not after: it starts listening the moment it is
    # asked to, and the MCP server's lifespan -- which is what would otherwise
    # open the databases -- does not run until `app.run()` below. A request
    # arriving in between would find a session bound to no engine.
    service.ensure_initialized()

    rest_server = None
    if settings.REST_ENABLED:
        from .rest_api import serve_in_thread

        rest_server = serve_in_thread()

    try:
        app.run(transport="http", host=settings.HOST, port=settings.PORT)
    finally:
        if rest_server is not None:
            rest_server.stop()


if __name__ == "__main__":
    serve()
