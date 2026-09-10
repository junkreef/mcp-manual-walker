"""Starting the server before its database exists.

`docker compose up` brings the server up next to an empty Qdrant, and the
corpus arrives afterwards through `db_manager import`. That ordering is normal
rather than exceptional, so the server has to notice when the store appears
instead of complaining until somebody restarts it.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from mcp_manual_walker import config, main, service
from mcp_manual_walker.config import settings


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts from a server that has attached to nothing."""
    main.app_state.store = None
    main.app_state.embedder = None
    main.app_state.init_error = None
    yield
    main.app_state.store = None
    main.app_state.embedder = None
    main.app_state.init_error = None


def a_store(model=None):
    store = MagicMock()
    store.embedding_model = model or settings.EMBEDDING_MODEL
    return store


def test_a_missing_store_is_reported_with_the_reason_it_is_missing():
    with patch.object(
        service, "open_store", side_effect=RuntimeError("collection does not exist")
    ):
        with pytest.raises(ToolError, match="collection does not exist"):
            main._require_store()


def test_the_store_is_picked_up_once_it_appears_without_a_restart():
    """This is the whole point: `up`, then `import`, then just use it."""
    with patch.object(
        service, "open_store", side_effect=RuntimeError("collection does not exist")
    ):
        with pytest.raises(ToolError):
            main._require_store()

    store = a_store()
    with (
        patch.object(service, "open_store", return_value=store),
        patch.object(service, "get_embedder", return_value=MagicMock()),
    ):
        assert main._require_store() is store
    assert main.app_state.init_error is None


def test_reopening_does_not_reload_the_embedding_model():
    """It is the expensive half of startup and has nothing to do with the store."""
    embedder = MagicMock()
    main.app_state.embedder = embedder

    with (
        patch.object(service, "open_store", return_value=a_store()),
        patch.object(service, "get_embedder") as loader,
    ):
        main._require_store()

    loader.assert_not_called()
    assert main.app_state.embedder is embedder


def test_reopening_loads_the_model_if_startup_never_managed_to():
    with (
        patch.object(service, "open_store", return_value=a_store()),
        patch.object(service, "get_embedder") as loader,
    ):
        main._require_store()

    loader.assert_called_once()


def test_a_store_built_by_another_model_is_still_refused_on_reopen():
    """The check that catches a mismatch must not be skipped by this path."""
    with (
        patch.object(
            service,
            "open_store",
            return_value=a_store("intfloat/multilingual-e5-small"),
        ),
        patch.object(service, "get_embedder", return_value=MagicMock()),
    ):
        with pytest.raises(ToolError, match="multilingual-e5-small"):
            main._require_store()

    assert main.app_state.store is None


def test_an_attached_store_is_returned_without_asking_the_backend_again():
    store = a_store()
    main.app_state.store = store

    with patch.object(service, "open_store") as opener:
        assert main._require_store() is store

    opener.assert_not_called()


def test_init_records_the_failure_instead_of_raising():
    """The server has to start even with no database, so the tools can say why."""
    with patch.object(service, "open_store", side_effect=RuntimeError("no server")):
        service.init_vector_store()

    assert main.app_state.store is None
    assert "no server" in main.app_state.init_error


class _FakeRestServer:
    """Stands in for the threaded REST server, recording that it was stopped."""

    def __init__(self, order):
        self.order = order
        self.stopped = False

    def stop(self, timeout=10.0):
        self.stopped = True
        self.order.append("rest stopped")


def test_the_rest_api_is_not_listening_before_the_databases_are_open(monkeypatch):
    """It starts first and answers immediately, so it must not start first.

    `SessionLocal` is bound to no engine until `init_db` runs, and the MCP
    server's lifespan -- which used to be the only thing that ran it -- does
    not start until `app.run`. A request landing in between would fail with an
    error about a missing bind rather than with anything a caller could act on.
    """
    from mcp_manual_walker import rest_api

    order = []
    server = _FakeRestServer(order)

    monkeypatch.setattr(config.settings, "REST_ENABLED", True)
    monkeypatch.setattr(service, "ensure_initialized", lambda: order.append("init"))
    monkeypatch.setattr(
        rest_api, "serve_in_thread", lambda: (order.append("rest"), server)[1]
    )
    monkeypatch.setattr(main.app, "run", lambda **kwargs: order.append("mcp"))

    main.serve()

    assert order == ["init", "rest", "mcp", "rest stopped"]


def test_the_rest_half_is_stopped_even_when_the_mcp_half_fails(monkeypatch):
    from mcp_manual_walker import rest_api

    order = []
    server = _FakeRestServer(order)

    monkeypatch.setattr(config.settings, "REST_ENABLED", True)
    monkeypatch.setattr(service, "ensure_initialized", lambda: None)
    monkeypatch.setattr(rest_api, "serve_in_thread", lambda: server)
    monkeypatch.setattr(
        main.app, "run", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        main.serve()

    assert server.stopped


def test_no_rest_server_is_started_when_it_is_disabled(monkeypatch):
    from mcp_manual_walker import rest_api

    started = []

    monkeypatch.setattr(config.settings, "REST_ENABLED", False)
    monkeypatch.setattr(service, "ensure_initialized", lambda: None)
    monkeypatch.setattr(rest_api, "serve_in_thread", lambda: started.append(1))
    monkeypatch.setattr(main.app, "run", lambda **kwargs: None)

    main.serve()

    assert started == []


def test_the_databases_are_opened_once_per_process(monkeypatch):
    """Both servers ask; only the first should pay for the embedding model."""
    monkeypatch.setattr(service, "_initialized", False)
    calls = []
    monkeypatch.setattr(service, "init_vector_store", lambda: calls.append(1))
    monkeypatch.setattr(service, "init_db", lambda: None)

    service.ensure_initialized()
    service.ensure_initialized()

    assert calls == [1]
