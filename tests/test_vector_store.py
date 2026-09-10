"""The vector backend interface, and the Chroma implementation of it.

`test_api.py` already exercises the whole path against a real Chroma
collection. What is pinned here is the translation layer that a second backend
will have to reproduce exactly: the three filter shapes, the order `get()` is
obliged to return, the direction a score counts in, and the two cases where the
right answer is to make no call at all.
"""

from unittest.mock import MagicMock, patch

import pytest

from mcp_manual_walker.chroma_store import ChromaVectorStore, _where
from mcp_manual_walker.config import settings
from mcp_manual_walker.vector_store import (
    Chunk,
    ChunkFilter,
    batched,
    open_store,
    reset_store,
)


@pytest.fixture
def store():
    collection = MagicMock()
    return ChromaVectorStore(MagicMock(), collection), collection


# --- the filter -----------------------------------------------------------


def test_an_empty_filter_matches_everything():
    assert _where(ChunkFilter()) is None
    assert ChunkFilter().is_empty()


def test_one_condition_is_emitted_bare():
    """Chroma rejects a one-element $and, so it must not be wrapped."""
    assert _where(ChunkFilter(manual_ids=["m1"])) == {"manual_id": {"$in": ["m1"]}}
    assert _where(ChunkFilter(bookmark_ids=["b1", "b2"])) == {
        "bookmark_id": {"$in": ["b1", "b2"]}
    }


def test_two_conditions_are_joined_with_and():
    assert _where(ChunkFilter(manual_ids=["m1"], bookmark_ids=["b1"])) == {
        "$and": [{"manual_id": {"$in": ["m1"]}}, {"bookmark_id": {"$in": ["b1"]}}]
    }


def test_an_empty_bookmark_set_matches_nothing_rather_than_anything():
    """A section with no chunks must return none, not the whole manual."""
    assert ChunkFilter(manual_ids=["m1"], bookmark_ids=[]).matches_nothing()
    assert ChunkFilter(manual_ids=[]).matches_nothing()
    assert not ChunkFilter(manual_ids=["m1"]).matches_nothing()
    assert not ChunkFilter().matches_nothing()


def test_scrolling_an_impossible_filter_asks_the_backend_nothing(store):
    vector_store, collection = store
    assert list(
        vector_store.scroll(ChunkFilter(manual_ids=["m"], bookmark_ids=[]))
    ) == []
    collection.get.assert_not_called()


def test_searching_an_impossible_filter_asks_the_backend_nothing(store):
    vector_store, collection = store
    impossible = ChunkFilter(manual_ids=["m"], bookmark_ids=[])
    assert vector_store.search([0.1], 5, impossible) == []
    collection.query.assert_not_called()


# --- reads ----------------------------------------------------------------


def test_get_returns_chunks_in_the_order_asked_for(store):
    """Chroma does not promise an order; the fusion that calls this needs one."""
    vector_store, collection = store
    collection.get.return_value = {
        "ids": ["c2", "c0", "c1"],
        "documents": ["two", "zero", "one"],
        "metadatas": [{"n": 2}, {"n": 0}, {"n": 1}],
    }

    got = vector_store.get(["c0", "c1", "c2"])

    assert [c.id for c in got] == ["c0", "c1", "c2"]
    assert [c.document for c in got] == ["zero", "one", "two"]


def test_get_drops_ids_the_store_no_longer_holds(store):
    """The BM25 index can name a chunk a delete already took out."""
    vector_store, collection = store
    collection.get.return_value = {
        "ids": ["c0"],
        "documents": ["zero"],
        "metadatas": [{}],
    }

    assert [c.id for c in vector_store.get(["c0", "gone"])] == ["c0"]


def test_get_of_nothing_asks_the_backend_nothing(store):
    vector_store, collection = store
    assert vector_store.get([]) == []
    collection.get.assert_not_called()


def test_a_filtered_scroll_is_one_unpaged_read(store):
    """A filter over several manuals blows SQLite's parameter limit.

    The callers already work one manual (or one section) at a time, so the
    filtered path deliberately does not page.
    """
    vector_store, collection = store
    collection.get.return_value = {
        "ids": ["c0"],
        "documents": ["zero"],
        "metadatas": [{"manual_id": {"$in": ["m1"]}}],
    }

    chunks = list(vector_store.scroll(ChunkFilter(manual_ids=["m1"])))

    assert [c.id for c in chunks] == ["c0"]
    assert collection.get.call_count == 1
    assert collection.get.call_args.kwargs["where"] == {"manual_id": {"$in": ["m1"]}}


def test_an_unfiltered_scroll_pages_until_the_store_is_exhausted(store):
    """The two callers that ask for everything walk half a million chunks."""
    vector_store, collection = store
    collection.get.side_effect = [
        {"ids": ["a", "b"], "documents": ["a", "b"], "metadatas": [{}, {}]},
        {"ids": ["c"], "documents": ["c"], "metadatas": [{}]},
        {"ids": [], "documents": [], "metadatas": []},
    ]

    chunks = list(vector_store.scroll(ChunkFilter(), batch_size=2))

    assert [c.id for c in chunks] == ["a", "b", "c"]
    offsets = [call.kwargs["offset"] for call in collection.get.call_args_list]
    assert offsets == [0, 2, 3]


def test_a_scroll_asks_for_embeddings_only_when_wanted(store):
    vector_store, collection = store
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    list(vector_store.scroll(ChunkFilter(manual_ids=["m"])))
    assert "embeddings" not in collection.get.call_args.kwargs["include"]

    list(vector_store.scroll(ChunkFilter(manual_ids=["m"]), with_embedding=True))
    assert "embeddings" in collection.get.call_args.kwargs["include"]


def test_a_scroll_that_wants_no_document_still_returns_ids(store):
    vector_store, collection = store
    collection.get.return_value = {"ids": ["c0"], "metadatas": [{}]}

    chunks = list(
        vector_store.scroll(ChunkFilter(manual_ids=["m"]), with_document=False)
    )
    assert [c.id for c in chunks] == ["c0"]
    assert "documents" not in collection.get.call_args.kwargs["include"]


def test_numpy_embeddings_come_back_as_plain_lists(store):
    """The export serializes these to JSON, which cannot take an ndarray."""
    numpy = pytest.importorskip("numpy")
    vector_store, collection = store
    collection.get.return_value = {
        "ids": ["c0"],
        "documents": ["zero"],
        "metadatas": [{}],
        "embeddings": [numpy.array([0.1, 0.2], dtype=numpy.float32)],
    }

    chunk = next(
        vector_store.scroll(ChunkFilter(manual_ids=["m"]), with_embedding=True)
    )
    assert isinstance(chunk.embedding, list)
    assert chunk.embedding == pytest.approx([0.1, 0.2], abs=1e-6)


# --- search ---------------------------------------------------------------


def test_a_cosine_distance_is_reported_as_a_similarity(store):
    """Chroma counts down, Qdrant counts up; Hit.score always counts up."""
    vector_store, collection = store
    collection.query.return_value = {
        "ids": [["near", "far"]],
        "documents": [["a", "b"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.05, 0.60]],
    }

    hits = vector_store.search([0.1, 0.2], limit=2)

    assert [h.id for h in hits] == ["near", "far"]
    assert hits[0].score == pytest.approx(0.95)
    assert hits[1].score == pytest.approx(0.40)
    assert hits[0].score > hits[1].score


def test_search_passes_the_query_vector_and_never_a_text(store):
    """No backend is ever given an embedding function of its own."""
    vector_store, collection = store
    collection.query.return_value = {"ids": [[]]}

    vector_store.search([0.1, 0.2], limit=7, where=ChunkFilter(manual_ids=["m1"]))

    kwargs = collection.query.call_args.kwargs
    assert kwargs["query_embeddings"] == [[0.1, 0.2]]
    assert kwargs["n_results"] == 7
    assert kwargs["where"] == {"manual_id": {"$in": ["m1"]}}


def test_a_search_that_matches_nothing_returns_an_empty_list(store):
    vector_store, collection = store
    collection.query.return_value = {"ids": [[]]}
    assert vector_store.search([0.1], limit=5) == []


# --- writes ---------------------------------------------------------------


def test_add_slices_large_writes(store):
    vector_store, collection = store
    chunks = [Chunk(id=f"c{i}", document="x", embedding=[0.1]) for i in range(2500)]

    vector_store.add(chunks)

    sizes = [len(call.kwargs["ids"]) for call in collection.add.call_args_list]
    assert sizes == [1000, 1000, 500]
    assert sum(sizes) == 2500


def test_add_keeps_a_chunks_four_fields_together(store):
    vector_store, collection = store
    vector_store.add([Chunk(id="c0", document="text", metadata={"m": 1},
                            embedding=[0.5])])

    kwargs = collection.add.call_args.kwargs
    assert kwargs["ids"] == ["c0"]
    assert kwargs["documents"] == ["text"]
    assert kwargs["metadatas"] == [{"m": 1}]
    assert kwargs["embeddings"] == [[0.5]]


def test_delete_manual_names_the_manual(store):
    vector_store, collection = store
    vector_store.delete_manual("m1")
    collection.delete.assert_called_once_with(where={"manual_id": "m1"})


def test_count_is_forwarded(store):
    vector_store, collection = store
    collection.count.return_value = 42
    assert vector_store.count() == 42


def test_the_recorded_model_comes_from_the_collection_metadata(store):
    vector_store, collection = store
    collection.metadata = {"embedding_model": "Qwen/Qwen3-Embedding-0.6B"}
    assert vector_store.embedding_model == "Qwen/Qwen3-Embedding-0.6B"


def test_a_store_with_no_recorded_model_reports_none(store):
    vector_store, collection = store
    collection.metadata = None
    assert vector_store.embedding_model is None


# --- the factory ----------------------------------------------------------


def test_open_store_dispatches_on_the_configured_backend():
    with (
        patch.object(settings, "VECTOR_BACKEND", "chroma"),
        patch.object(ChromaVectorStore, "open") as opened,
    ):
        open_store()
    opened.assert_called_once_with(embedder=None, create=False)


def test_open_store_rejects_an_unknown_backend():
    with patch.object(settings, "VECTOR_BACKEND", "pinecone"):
        with pytest.raises(ValueError, match="expected 'chroma'"):
            open_store()


def test_reset_store_dispatches_on_the_configured_backend():
    with (
        patch.object(settings, "VECTOR_BACKEND", "chroma"),
        patch.object(ChromaVectorStore, "reset") as reset,
    ):
        reset_store()
    reset.assert_called_once()


def test_reset_store_rejects_an_unknown_backend():
    with patch.object(settings, "VECTOR_BACKEND", "pinecone"):
        with pytest.raises(ValueError, match="expected 'chroma'"):
            reset_store()


def test_reset_removes_the_store_directory(tmp_path):
    target = tmp_path / "chroma_db"
    target.mkdir()
    (target / "chroma.sqlite3").write_text("x")

    with patch.object(settings, "CHROMADB_PATH", target):
        ChromaVectorStore.reset()

    assert not target.exists()


def test_resetting_a_store_that_is_not_there_is_not_an_error(tmp_path):
    with patch.object(settings, "CHROMADB_PATH", tmp_path / "absent"):
        ChromaVectorStore.reset()


# --- helpers --------------------------------------------------------------


def test_batched_groups_without_losing_anything():
    chunks = [Chunk(id=str(i)) for i in range(7)]
    groups = list(batched(chunks, 3))
    assert [len(g) for g in groups] == [3, 3, 1]
    assert [c.id for g in groups for c in g] == [str(i) for i in range(7)]


def test_batched_of_nothing_yields_nothing():
    assert list(batched([], 3)) == []
