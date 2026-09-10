"""The REST search API, served next to the MCP endpoint on its own port.

MCP is how an agent uses this corpus, and it is the only way an agent should.
It is not the only kind of caller: a RAG pipeline retrieving passages for a
prompt, a script checking what a manual says about an error code, and the test
console in `webgui/` all want to ask the same question over plain HTTP and get
JSON back. That is what this is -- the same retrieval the `search_manual` tool
performs, reachable without a session, a handshake or a tool call.

Two things it does that the MCP tool does not:

* **It searches the whole corpus by default.** `manual_id` and `bookmark_id`
  narrow a search here rather than being required to start one. An agent has
  browsed its way to a manual before it searches; a RAG client has a question
  and nothing else, and making it choose a manual first would be asking it to
  solve retrieval before it may use retrieval.
* **It answers a browser.** CORS is on, so a page served from anywhere may
  call this, and the figure endpoint returns an image a `<img src>` can point
  at directly.

Everything below is a thin translation of `service.py`: parse the request,
call the service, turn a `ServiceError` into a status code. No retrieval logic
lives here, so the API and the MCP tools cannot drift apart.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Annotated, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import service
from .config import settings
from .database import SessionLocal, init_db
from .models import Manual
from .schemas import (
    DirectoryEntry,
    FigureInfo,
    HealthResponse,
    ManualMetadata,
    MarkdownContent,
    SearchResponse,
)
from .service import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    ServiceError,
)

logger = logging.getLogger(__name__)

# How a service failure is reported over HTTP. A manual that does not exist and
# a vector store that has not been imported yet are both refusals, but only one
# of them is worth retrying, and a client can only tell from the status.
_STATUS_FOR_KIND = {
    "not_found": 404,
    "invalid": 400,
    "unavailable": 503,
    "internal": 500,
}


class SearchRequest(BaseModel):
    """A search, as a JSON body."""

    query: str = Field(
        ...,
        description="The question or text to search for.",
        examples=["how do I mount a zFS file system"],
    )
    manual_id: Optional[str] = Field(
        None,
        description=(
            "Restrict the search to one manual. Omit to search the whole "
            "corpus, which is the point of this endpoint."
        ),
    )
    bookmark_id: Optional[str] = Field(
        None,
        description=(
            "Restrict the search to one section and its subsections. The "
            "manual is implied, so `manual_id` may be omitted alongside it."
        ),
    )
    limit: int = Field(
        DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="How many results to return, best first.",
    )


def _http_error(error: ServiceError) -> HTTPException:
    """Renders a service failure as the status code that describes it."""
    return HTTPException(
        status_code=_STATUS_FOR_KIND.get(error.kind, 500), detail=str(error)
    )


def require_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    authorization: Annotated[Optional[str], Header()] = None,
) -> None:
    """Checks the key, when there is one to check.

    An unset `REST_API_KEY` leaves the API open, which is what a loopback bind
    or a private compose network already assumes. Setting one closes it without
    any other change, and both header spellings are accepted because a browser
    reaches for `X-API-Key` and everything else reaches for `Authorization`.
    """
    expected = settings.REST_API_KEY.strip()
    if not expected:
        return

    presented = x_api_key
    if not presented and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = value.strip()

    if presented != expected:
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required, as X-API-Key or a bearer token.",
        )


def create_app() -> FastAPI:
    """Builds the REST application.

    A factory rather than a module-level object so that a test can build one
    against whatever settings it has just monkeypatched, the way the CORS
    origins and the API key both need.
    """
    api = FastAPI(
        title="MCP Manual Walker search API",
        description=(
            "Hybrid (vector + BM25) retrieval over a corpus of PDF manuals. "
            "The same index the MCP server serves, over plain HTTP."
        ),
        version="1.0.0",
    )

    origins = [o.strip() for o in settings.REST_CORS_ORIGINS.split(",") if o.strip()]
    if origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            # The API key travels as a header, so the browser has to be allowed
            # to send one it did not think of itself.
            allow_headers=["*"],
        )

    guarded = [Depends(require_api_key)]

    @api.get(
        "/health",
        response_model=HealthResponse,
        summary="Whether a search can be answered",
        tags=["status"],
    )
    def health() -> HealthResponse:
        """Reports readiness.

        Deliberately outside the API key check: this is what a container health
        check and a load balancer call, and neither should have to hold a
        credential to ask whether the process is alive. It reveals only sizes
        and the model name, never content.
        """
        available, detail = service.store_status()

        manuals: Optional[int] = None
        db = SessionLocal()
        try:
            manuals = db.scalar(select(func.count()).select_from(Manual))
        except Exception as e:  # noqa: BLE001 - a health check never fails hard
            logger.warning(f"Health check could not count the manuals: {e}")
        finally:
            db.close()

        chunks: Optional[int] = None
        if available and service.app_state.store is not None:
            try:
                chunks = service.app_state.store.count()
            except Exception as e:  # noqa: BLE001 - as above
                logger.warning(f"Health check could not count the chunks: {e}")

        return HealthResponse(
            status="ok" if available else "degraded",
            vector_backend=settings.VECTOR_BACKEND,
            embedding_model=settings.EMBEDDING_MODEL,
            manuals=manuals,
            chunks=chunks,
            detail=detail,
        )

    @api.post(
        "/search",
        response_model=SearchResponse,
        dependencies=guarded,
        summary="Search the corpus",
        tags=["search"],
    )
    def search(request: SearchRequest) -> SearchResponse:
        """Returns the chunks best matching a query, best first."""
        try:
            hits = service.search(
                request.query,
                manual_id=request.manual_id,
                bookmark_id=request.bookmark_id,
                limit=request.limit,
            )
        except ServiceError as e:
            raise _http_error(e) from e

        return SearchResponse(
            query=request.query,
            manual_id=request.manual_id,
            bookmark_id=request.bookmark_id,
            limit=request.limit,
            count=len(hits),
            results=hits,
        )

    @api.get(
        "/search",
        response_model=SearchResponse,
        dependencies=guarded,
        summary="Search the corpus from a query string",
        tags=["search"],
    )
    def search_via_query_string(
        q: Annotated[str, Query(description="The question or text to search for.")],
        manual_id: Annotated[Optional[str], Query()] = None,
        bookmark_id: Annotated[Optional[str], Query()] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_SEARCH_LIMIT)] = DEFAULT_SEARCH_LIMIT,
    ) -> SearchResponse:
        """The same search as a GET, so a browser or `curl` can run one."""
        return search(
            SearchRequest(
                query=q, manual_id=manual_id, bookmark_id=bookmark_id, limit=limit
            )
        )

    @api.get(
        "/manuals",
        response_model=List[DirectoryEntry],
        dependencies=guarded,
        summary="List one folder of the manual library",
        tags=["library"],
    )
    def manuals(
        folder: Annotated[
            str,
            Query(
                description=(
                    "The folder to list, relative to the library root. Empty "
                    "lists the root. Nothing deeper than one level is returned."
                )
            ),
        ] = "",
    ) -> List[DirectoryEntry]:
        """Returns the manuals and subfolders directly inside a folder."""
        try:
            return service.list_folder(folder)
        except ServiceError as e:
            raise _http_error(e) from e

    @api.get(
        "/manuals/{manual_id}",
        response_model=ManualMetadata,
        dependencies=guarded,
        summary="Metadata and table of contents of one manual",
        tags=["library"],
    )
    def manual(manual_id: str) -> ManualMetadata:
        """Returns a manual's metadata and its hierarchical table of contents."""
        try:
            return service.manual_metadata(manual_id)
        except ServiceError as e:
            raise _http_error(e) from e

    @api.get(
        "/bookmarks/{bookmark_id}/markdown",
        response_model=MarkdownContent,
        dependencies=guarded,
        summary="The text of one section",
        tags=["content"],
    )
    def markdown(bookmark_id: str) -> MarkdownContent:
        """Returns the Markdown of a section and everything nested under it."""
        try:
            return service.markdown_content(bookmark_id)
        except ServiceError as e:
            raise _http_error(e) from e

    @api.get(
        "/figures/{figure_id}",
        response_model=FigureInfo,
        dependencies=guarded,
        summary="What a figure is",
        tags=["content"],
    )
    def figure(figure_id: str) -> FigureInfo:
        """Returns a figure's caption, labels, description and size."""
        try:
            _, info = service.get_figure(figure_id)
        except ServiceError as e:
            raise _http_error(e) from e
        return info

    @api.get(
        "/figures/{figure_id}/image",
        dependencies=guarded,
        summary="The image of a figure",
        tags=["content"],
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}},
    )
    def figure_image(figure_id: str) -> Response:
        """Returns the figure itself, as the bytes stored at build time.

        Separate from the metadata endpoint so that a page can put the id in an
        `<img src>` and let the browser do the fetching, rather than having to
        decode a base64 field out of a JSON document first.
        """
        try:
            image, info = service.get_figure(figure_id)
        except ServiceError as e:
            raise _http_error(e) from e

        return Response(
            content=image,
            media_type=info.mime_type or "image/png",
            # A figure is a build artefact: it cannot change without the manual
            # being rebuilt, and a rebuild gives it a new id.
            headers={"Cache-Control": "public, max-age=86400"},
        )

    return api


class RestServer:
    """The REST API running on a thread of its own, and the handle to stop it."""

    def __init__(self, server: uvicorn.Server, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread

    @property
    def started(self) -> bool:
        return self._server.started

    def stop(self, timeout: float = 10.0) -> None:
        """Asks the server to finish what it is serving and waits for it."""
        self._server.should_exit = True
        self._thread.join(timeout=timeout)


def serve_in_thread(
    app: Optional[FastAPI] = None, startup_timeout: float = 30.0
) -> RestServer:
    """Starts the REST API on a background thread and returns once it listens.

    The thread is what keeps this compatible with the MCP server sharing the
    process: uvicorn installs SIGINT and SIGTERM handlers only when it is
    started on the main thread, so the copy running here installs none and the
    signal keeps reaching the MCP server, which is the one that should decide
    when the process ends.
    """
    config = uvicorn.Config(
        app or create_app(),
        host=settings.REST_HOST,
        port=settings.REST_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        # The application state -- the vector store and the embedding model --
        # is set up by whoever started this process; there is nothing for a
        # second lifespan to do, and running one would re-open the store.
        lifespan="off",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="rest-api", daemon=True)
    thread.start()

    # Wait for the socket rather than for the thread: a port that is not yet
    # listening when this returns is a race every caller would have to repeat.
    deadline = time.monotonic() + startup_timeout
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)

    if not server.started:
        raise RuntimeError(
            f"The REST API failed to start on "
            f"{settings.REST_HOST}:{settings.REST_PORT}."
        )

    logger.info(
        f"REST search API listening on "
        f"http://{settings.REST_HOST}:{settings.REST_PORT}"
    )
    return RestServer(server, thread)


def main() -> None:
    """Runs the REST API on its own, without the MCP server beside it."""
    logging.basicConfig(level=settings.LOG_LEVEL)
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    service.init_vector_store()
    uvicorn.run(
        create_app(),
        host=settings.REST_HOST,
        port=settings.REST_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
