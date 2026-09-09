"""Starting the server before its database exists.

`docker compose up` brings the server up next to an empty Qdrant, and the
corpus arrives afterwards through `db_manager import`. That ordering is normal
rather than exceptional, so the server has to notice when the store appears
instead of complaining until somebody restarts it.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from mcp_manual_walker import main
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
        main, "open_store", side_effect=RuntimeError("collection does not exist")
    ):
        with pytest.raises(ToolError, match="collection does not exist"):
            main._require_store()


def test_the_store_is_picked_up_once_it_appears_without_a_restart():
    """This is the whole point: `up`, then `import`, then just use it."""
    with patch.object(
        main, "open_store", side_effect=RuntimeError("collection does not exist")
    ):
        with pytest.raises(ToolError):
            main._require_store()

    store = a_store()
    with (
        patch.object(main, "open_store", return_value=store),
        patch.object(main, "get_embedder", return_value=MagicMock()),
    ):
        assert main._require_store() is store
    assert main.app_state.init_error is None


def test_reopening_does_not_reload_the_embedding_model():
    """It is the expensive half of startup and has nothing to do with the store."""
    embedder = MagicMock()
    main.app_state.embedder = embedder

    with (
        patch.object(main, "open_store", return_value=a_store()),
        patch.object(main, "get_embedder") as loader,
    ):
        main._require_store()

    loader.assert_not_called()
    assert main.app_state.embedder is embedder


def test_reopening_loads_the_model_if_startup_never_managed_to():
    with (
        patch.object(main, "open_store", return_value=a_store()),
        patch.object(main, "get_embedder") as loader,
    ):
        main._require_store()

    loader.assert_called_once()


def test_a_store_built_by_another_model_is_still_refused_on_reopen():
    """The check that catches a mismatch must not be skipped by this path."""
    with (
        patch.object(
            main, "open_store", return_value=a_store("intfloat/multilingual-e5-small")
        ),
        patch.object(main, "get_embedder", return_value=MagicMock()),
    ):
        with pytest.raises(ToolError, match="multilingual-e5-small"):
            main._require_store()

    assert main.app_state.store is None


def test_an_attached_store_is_returned_without_asking_the_backend_again():
    store = a_store()
    main.app_state.store = store

    with patch.object(main, "open_store") as opener:
        assert main._require_store() is store

    opener.assert_not_called()


def test_init_records_the_failure_instead_of_raising():
    """The server has to start even with no database, so the tools can say why."""
    with patch.object(main, "open_store", side_effect=RuntimeError("no server")):
        main.init_vector_store()

    assert main.app_state.store is None
    assert "no server" in main.app_state.init_error
