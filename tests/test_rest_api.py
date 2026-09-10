"""The REST API, against the same fixture environment the MCP tools use.

Both front ends now sit on one service layer, so what is worth testing here is
not retrieval again -- `test_api.py` covers that through the tools -- but the
translation: corpus-wide search, status codes, the API key, CORS, the image
bytes, and the fact that the second port really does come up next to the first.
"""

import socket
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from test_api import (  # the fixture builder is shared rather than duplicated
    FIGURE_CAPTION,
    _build_test_environment,
    _test_state,
)

from mcp_manual_walker import config, database, rest_api, service


@pytest.fixture
def api(tmp_path, monkeypatch, dummy_pdf_factory):
    """A test client on the REST app, over the shared dummy corpus."""
    _build_test_environment(
        tmp_path, monkeypatch, dummy_pdf_factory, config.settings.EMBEDDING_MODEL
    )
    monkeypatch.setattr(config.settings, "REST_API_KEY", "")
    monkeypatch.setattr(config.settings, "REST_CORS_ORIGINS", "*")

    with TestClient(rest_api.create_app()) as client:
        yield client

    if database.engine:
        database.engine.dispose()


def test_health_reports_what_the_server_is_holding(api):
    body = api.get("/health").json()

    assert body["status"] == "ok"
    assert body["embedding_model"] == config.settings.EMBEDDING_MODEL
    assert body["manuals"] == 1
    # 30 page chunks plus the figure chunk.
    assert body["chunks"] == 31


def test_a_search_defaults_to_the_whole_corpus(api):
    """The whole reason this API exists: a question, and nothing else."""
    response = api.post("/search", json={"query": "unique text for search"})
    assert response.status_code == 200

    body = response.json()
    assert body["count"] > 0
    assert body["manual_path"] == "*"

    first = body["results"][0]
    assert first["rank"] == 1
    assert first["chunk_id"]
    assert first["score"] is not None
    assert first["retrieval"] in {"dense", "lexical", "both"}
    # A corpus-wide hit has to say which manual it came from.
    assert first["manual"]["file_name"] == "dummy_manual.pdf"
    assert first["manual_id"] == _test_state["manual_id"]
    assert first["manual_path"] == _test_state["manual_path"]


def test_the_same_search_can_be_run_as_a_get(api):
    """So that a browser address bar and `curl` are enough to try one."""
    body = api.get("/search", params={"q": "Content for page 20", "limit": 3}).json()

    assert body["limit"] == 3
    assert 0 < body["count"] <= 3
    assert "Content for page 20" in body["results"][0]["context"]


def test_a_search_can_use_a_path_returned_by_the_manuals_endpoint(api):
    entry = api.get("/manuals").json()[0]
    body = api.post(
        "/search",
        json={"query": "unique text", "manual_path": entry["path"]},
    ).json()

    assert body["manual_path"] == entry["path"]
    assert all(hit["manual_path"] == entry["path"] for hit in body["results"])


def test_a_search_accepts_a_wildcard_path(api):
    body = api.post(
        "/search",
        json={"query": "unique text", "manual_path": "dummy*"},
    ).json()

    assert body["count"] > 0
    assert body["manual_path"] == "dummy*"


def test_a_search_rejects_a_path_that_matches_no_manual(api):
    response = api.post(
        "/search",
        json={"query": "unique text", "manual_path": "missing/*"},
    )

    assert response.status_code == 404


def test_a_bookmark_can_narrow_a_corpus_search_to_a_section(api):
    """The default path range contains the bookmark's manual."""
    bookmark_id = _test_state["figure_bookmark_id"]
    body = api.post(
        "/search", json={"query": "unique text", "bookmark_id": bookmark_id}
    ).json()

    assert body["count"] > 0
    assert all(hit["bookmark_id"] == bookmark_id for hit in body["results"])


def test_a_figure_hit_carries_the_figure_it_describes(api):
    body = api.post("/search", json={"query": FIGURE_CAPTION, "limit": 10}).json()

    figure_hits = [hit for hit in body["results"] if hit["chunk_type"] == "figure"]
    assert figure_hits, "the figure chunk should be findable by its caption"
    assert figure_hits[0]["figure"]["id"] == _test_state["figure_id"]


def test_an_empty_query_is_refused_rather_than_answered(api):
    assert api.post("/search", json={"query": "   "}).status_code == 400


def test_a_limit_outside_the_allowed_range_is_refused(api):
    assert api.post("/search", json={"query": "x", "limit": 0}).status_code == 422
    assert (
        api.post(
            "/search", json={"query": "x", "limit": service.MAX_SEARCH_LIMIT + 1}
        ).status_code
        == 422
    )


def test_the_library_is_browsed_one_folder_at_a_time(api):
    entries = api.get("/manuals").json()

    assert [e["name"] for e in entries] == ["dummy_manual.pdf"]
    assert entries[0]["id"] == _test_state["manual_id"]


def test_an_unknown_folder_is_a_404(api):
    assert api.get("/manuals", params={"folder": "nope"}).status_code == 404


def test_a_manual_answers_with_its_table_of_contents(api):
    body = api.get(f"/manuals/{_test_state['manual_id']}").json()

    assert body["file_name"] == "dummy_manual.pdf"
    assert [chapter["title"] for chapter in body["table_of_contents"]] == [
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
    ]


def test_an_unknown_manual_is_a_404(api):
    assert api.get("/manuals/nope").status_code == 404


def test_a_section_answers_with_its_markdown(api):
    bookmark_id = _test_state["figure_bookmark_id"]
    body = api.get(f"/bookmarks/{bookmark_id}/markdown").json()

    assert "Content for page 2." in body["markdown_content"]
    assert body["figures"][0]["id"] == _test_state["figure_id"]


def test_an_unknown_bookmark_is_a_404(api):
    assert api.get("/bookmarks/nope/markdown").status_code == 404


def test_a_figure_answers_with_its_metadata_and_its_bytes(api):
    figure_id = _test_state["figure_id"]

    info = api.get(f"/figures/{figure_id}").json()
    assert info["caption"] == FIGURE_CAPTION
    assert info["manual_id"] == _test_state["manual_id"]

    image = api.get(f"/figures/{figure_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    # The bytes are the ones stored at build time, not a re-encoding.
    assert image.content == _test_state["figure_png"]


def test_an_unknown_figure_is_a_404(api):
    assert api.get("/figures/nope/image").status_code == 404


def test_a_search_without_a_vector_store_is_a_503_that_says_so(api, monkeypatch):
    """The corpus arrives after the server does, and 503 is the honest answer."""
    def missing(*args, **kwargs):
        raise RuntimeError("no collection")

    monkeypatch.setattr(service.app_state, "store", None)
    monkeypatch.setattr(service, "open_store", missing)

    response = api.post("/search", json={"query": "anything"})
    assert response.status_code == 503
    assert "no collection" in response.json()["detail"]


def test_a_named_origin_is_allowed_to_call_the_api(
    tmp_path, monkeypatch, dummy_pdf_factory
):
    """The console calls the API from a browser, so its origin has to pass."""
    _build_test_environment(
        tmp_path, monkeypatch, dummy_pdf_factory, config.settings.EMBEDDING_MODEL
    )
    monkeypatch.setattr(
        config.settings, "REST_CORS_ORIGINS", "http://localhost:8080,http://box:9000"
    )

    with TestClient(rest_api.create_app()) as client:
        allowed = client.get("/health", headers={"Origin": "http://localhost:8080"})
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:8080"

        # Any other page the user happens to have open must not be able to read
        # the corpus off their own machine.
        refused = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in refused.headers

    if database.engine:
        database.engine.dispose()


def test_no_origins_means_no_cors_headers(tmp_path, monkeypatch, dummy_pdf_factory):
    """Which is all a same-origin caller, or one that is not a browser, needs."""
    _build_test_environment(
        tmp_path, monkeypatch, dummy_pdf_factory, config.settings.EMBEDDING_MODEL
    )
    monkeypatch.setattr(config.settings, "REST_CORS_ORIGINS", "")

    with TestClient(rest_api.create_app()) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:8080"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    if database.engine:
        database.engine.dispose()


def test_without_a_key_configured_the_api_is_open(api):
    assert api.post("/search", json={"query": "unique"}).status_code == 200


def test_a_configured_key_is_required(api, monkeypatch):
    monkeypatch.setattr(config.settings, "REST_API_KEY", "s3cret")

    assert api.post("/search", json={"query": "unique"}).status_code == 401
    assert (
        api.post(
            "/search", json={"query": "unique"}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        api.post(
            "/search", json={"query": "unique"}, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )
    assert (
        api.post(
            "/search",
            json={"query": "unique"},
            headers={"Authorization": "Bearer s3cret"},
        ).status_code
        == 200
    )


def test_health_is_reachable_without_the_key(api, monkeypatch):
    """A container health check holds no credentials and needs none."""
    monkeypatch.setattr(config.settings, "REST_API_KEY", "s3cret")
    assert api.get("/health").status_code == 200


def test_the_api_really_listens_on_its_own_port(
    tmp_path, monkeypatch, dummy_pdf_factory
):
    """The second port is the feature; a client has to be able to reach it."""
    _build_test_environment(
        tmp_path, monkeypatch, dummy_pdf_factory, config.settings.EMBEDDING_MODEL
    )
    monkeypatch.setattr(config.settings, "REST_API_KEY", "")
    monkeypatch.setattr(config.settings, "REST_HOST", "127.0.0.1")
    # 0 asks the kernel for a free port, so the suite cannot collide with
    # whatever else is running on this machine.
    monkeypatch.setattr(config.settings, "REST_PORT", 0)

    server = rest_api.serve_in_thread()
    try:
        assert server.started
        port = server._server.servers[0].sockets[0].getsockname()[1]

        response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=10)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        found = httpx.post(
            f"http://127.0.0.1:{port}/search",
            json={"query": "unique text for search"},
            timeout=30,
        )
        assert found.status_code == 200
        assert found.json()["count"] > 0
    finally:
        server.stop()

    assert not any(t.name == "rest-api" for t in threading.enumerate())

    if database.engine:
        database.engine.dispose()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_taken_rest_port_fails_loudly_instead_of_serving_half_a_server(monkeypatch):
    """Moving REST_PORT onto something already listening must not be quiet.

    The alternative is a process that serves MCP, answers nothing on the REST
    port and reports no error, which is the hardest kind of misconfiguration to
    find: everything looks up until a client asks.

    The uvicorn thread raises the bind error on its way out, which is the
    mechanism being tested rather than an unhandled failure.
    """
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken = blocker.getsockname()[1]

    monkeypatch.setattr(config.settings, "REST_HOST", "127.0.0.1")
    monkeypatch.setattr(config.settings, "REST_PORT", taken)

    try:
        with pytest.raises(RuntimeError, match=f"127.0.0.1:{taken}"):
            rest_api.serve_in_thread(startup_timeout=10)
    finally:
        blocker.close()

    assert not any(t.name == "rest-api" and t.is_alive() for t in threading.enumerate())
