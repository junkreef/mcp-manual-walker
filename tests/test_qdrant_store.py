"""The Qdrant backend: the contract, and a live round trip.

Most of this runs against a mocked client, because what has to be pinned is the
translation -- ids, filters, payload shape, score direction -- and a server
cannot make those more true. The integration tests at the bottom are the ones
that can only be answered by a real Qdrant, and they skip themselves when there
is none: that the filters actually match, that a deterministic point id really
does overwrite rather than duplicate, and that a chunk survives the round trip
byte for byte, which is what the export archive depends on.

Run them with a server up:

    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    QDRANT_TEST_URL=http://localhost:6333 pytest tests/test_qdrant_store.py
"""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from mcp_manual_walker.config import settings
from mcp_manual_walker.qdrant_store import (
    CHUNK_ID_NAMESPACE,
    QdrantVectorStore,
    _filter,
    point_id,
)
from mcp_manual_walker.vector_store import Chunk, ChunkFilter

COLLECTION = "test_manual_chunks"


@pytest.fixture
def store():
    client = MagicMock()
    vector_store = QdrantVectorStore(client, COLLECTION, 4)
    vector_store._ready = True
    return vector_store, client


# --- point ids ------------------------------------------------------------


def test_a_chunk_id_maps_to_a_uuid_qdrant_will_accept():
    """Qdrant takes unsigned integers or UUIDs; this application's ids are
    strings like "<manual_id>_<n>"."""
    generated = point_id("f47ac10b-58cc-4372-a567-0e02b2c3d479_1081")
    assert uuid.UUID(generated).version == 5


def test_the_point_id_is_reproducible_from_the_chunk_id_alone():
    """A random namespace would make a re-import write second copies of every
    chunk instead of overwriting the first."""
    chunk_id = "manual-a_7"
    assert point_id(chunk_id) == point_id(chunk_id)
    assert point_id(chunk_id) == str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def test_different_chunks_get_different_points():
    assert point_id("manual-a_7") != point_id("manual-a_8")
    assert point_id("manual-a_7") != point_id("manual-b_7")


# --- filters --------------------------------------------------------------


def test_an_empty_filter_is_no_filter():
    assert _filter(ChunkFilter()) is None
    assert _filter(None) is None


def test_a_manual_filter_is_a_single_match():
    built = _filter(ChunkFilter(manual_ids=["m1"]))
    assert built.must == [
        models.FieldCondition(key="manual_id", match=models.MatchAny(any=["m1"]))
    ]


def test_a_bookmark_set_is_a_match_any():
    built = _filter(ChunkFilter(bookmark_ids=["b1", "b2"]))
    assert built.must == [
        models.FieldCondition(
            key="bookmark_id", match=models.MatchAny(any=["b1", "b2"])
        )
    ]


def test_both_conditions_are_required_together():
    built = _filter(ChunkFilter(manual_ids=["m1"], bookmark_ids=["b1"]))
    assert len(built.must) == 2


def test_an_impossible_filter_asks_the_server_nothing(store):
    vector_store, client = store
    impossible = ChunkFilter(manual_ids=["m"], bookmark_ids=[])

    assert list(vector_store.scroll(impossible)) == []
    assert vector_store.search([0.1], 5, impossible) == []
    client.scroll.assert_not_called()
    client.query_points.assert_not_called()


# --- payload shape --------------------------------------------------------


def test_a_chunk_travels_with_its_id_and_text_in_the_payload(store):
    vector_store, client = store
    vector_store.add(
        [Chunk(id="c0", document="text", metadata={"manual_id": "m1", "page": 3},
               embedding=[0.1, 0.2, 0.3, 0.4])]
    )

    point = client.upsert.call_args.kwargs["points"][0]
    assert point.id == point_id("c0")
    assert point.vector == [0.1, 0.2, 0.3, 0.4]
    assert point.payload == {
        "manual_id": "m1",
        "page": 3,
        "chunk_id": "c0",
        "document": "text",
    }


def test_the_reserved_payload_keys_are_stripped_on_the_way_out(store):
    """A chunk read back must equal the chunk written; the export depends on it."""
    vector_store, client = store
    record = MagicMock()
    record.payload = {
        "manual_id": "m1",
        "page": 3,
        "chunk_id": "c0",
        "document": "text",
    }
    record.vector = None
    client.scroll.return_value = ([record], None)

    chunk = next(vector_store.scroll(ChunkFilter(manual_ids=["m1"])))

    assert chunk.id == "c0"
    assert chunk.document == "text"
    assert chunk.metadata == {"manual_id": "m1", "page": 3}


def test_add_slices_large_writes(store):
    vector_store, client = store
    vector_store.add(
        [Chunk(id=f"c{i}", document="x", embedding=[0.1] * 4) for i in range(2500)]
    )
    sizes = [len(c.kwargs["points"]) for c in client.upsert.call_args_list]
    assert sizes == [1000, 1000, 500]


def test_adding_nothing_asks_the_server_nothing(store):
    vector_store, client = store
    vector_store.add([])
    client.upsert.assert_not_called()


# --- reads ----------------------------------------------------------------


def test_get_returns_chunks_in_the_order_asked_for(store):
    vector_store, client = store

    def record(chunk_id):
        found = MagicMock()
        found.payload = {"chunk_id": chunk_id, "document": chunk_id}
        return found

    # Deliberately out of order, as retrieve() is entitled to be.
    client.retrieve.return_value = [record("c2"), record("c0"), record("c1")]

    got = vector_store.get(["c0", "c1", "c2"])

    assert [c.id for c in got] == ["c0", "c1", "c2"]
    assert client.retrieve.call_args.kwargs["ids"] == [
        point_id("c0"), point_id("c1"), point_id("c2")
    ]


def test_get_drops_ids_the_store_no_longer_holds(store):
    vector_store, client = store
    found = MagicMock()
    found.payload = {"chunk_id": "c0", "document": "zero"}
    client.retrieve.return_value = [found]

    assert [c.id for c in vector_store.get(["c0", "gone"])] == ["c0"]


def test_get_of_nothing_asks_the_server_nothing(store):
    vector_store, client = store
    assert vector_store.get([]) == []
    client.retrieve.assert_not_called()


def test_a_scroll_follows_the_cursor_to_the_end(store):
    """Qdrant pages by point id, not by a numeric offset."""
    vector_store, client = store

    def record(chunk_id):
        found = MagicMock()
        found.payload = {"chunk_id": chunk_id, "document": chunk_id}
        found.vector = None
        return found

    client.scroll.side_effect = [
        ([record("a"), record("b")], "cursor-1"),
        ([record("c")], None),
    ]

    chunks = list(vector_store.scroll(ChunkFilter(), batch_size=2))

    assert [c.id for c in chunks] == ["a", "b", "c"]
    offsets = [call.kwargs["offset"] for call in client.scroll.call_args_list]
    assert offsets == [None, "cursor-1"]


def test_a_scroll_that_wants_no_text_leaves_it_on_the_server(store):
    """The text is the largest thing in a payload and the caller only wants ids."""
    vector_store, client = store
    client.scroll.return_value = ([], None)

    list(vector_store.scroll(ChunkFilter(manual_ids=["m"]), with_document=False))

    selector = client.scroll.call_args.kwargs["with_payload"]
    assert isinstance(selector, models.PayloadSelectorExclude)
    assert selector.exclude == ["document"]


def test_a_scroll_asks_for_vectors_only_when_wanted(store):
    vector_store, client = store
    client.scroll.return_value = ([], None)

    list(vector_store.scroll(ChunkFilter(manual_ids=["m"])))
    assert client.scroll.call_args.kwargs["with_vectors"] is False

    list(vector_store.scroll(ChunkFilter(manual_ids=["m"]), with_embedding=True))
    assert client.scroll.call_args.kwargs["with_vectors"] is True


# --- search ---------------------------------------------------------------


def test_a_qdrant_score_is_already_a_similarity(store):
    """Chroma counts down and Qdrant counts up; Hit.score always counts up."""
    vector_store, client = store

    def point(chunk_id, score):
        found = MagicMock()
        found.payload = {"chunk_id": chunk_id, "document": chunk_id}
        found.score = score
        return found

    response = MagicMock()
    response.points = [point("near", 0.95), point("far", 0.40)]
    client.query_points.return_value = response

    hits = vector_store.search([0.1] * 4, limit=2)

    assert [h.id for h in hits] == ["near", "far"]
    assert hits[0].score == pytest.approx(0.95)
    assert hits[0].score > hits[1].score


def test_search_passes_the_query_vector_the_limit_and_the_filter(store):
    vector_store, client = store
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response

    vector_store.search([0.1, 0.2], limit=7, where=ChunkFilter(manual_ids=["m1"]))

    kwargs = client.query_points.call_args.kwargs
    assert kwargs["query"] == [0.1, 0.2]
    assert kwargs["limit"] == 7
    assert kwargs["query_filter"].must[0].match.any == ["m1"]


def test_an_unquantized_search_asks_for_no_rescoring(store):
    vector_store, client = store
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response

    with patch.object(settings, "QDRANT_QUANTIZATION", "none"):
        vector_store.search([0.1] * 4, limit=5)

    assert client.query_points.call_args.kwargs["search_params"].quantization is None


def test_a_quantized_search_always_rescores(store):
    """A quantized shortlist ranked without rescoring is the quantizer's
    ranking, not the model's."""
    vector_store, client = store
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response

    with patch.object(settings, "QDRANT_QUANTIZATION", "int8"):
        vector_store.search([0.1] * 4, limit=5)

    quantization = client.query_points.call_args.kwargs["search_params"].quantization
    assert quantization.rescore is True


def test_an_unknown_quantization_is_refused(store):
    vector_store, _ = store
    with patch.object(settings, "QDRANT_QUANTIZATION", "float4"):
        with pytest.raises(ValueError, match="expected 'none', 'int8' or 'binary'"):
            vector_store.search([0.1] * 4, limit=5)


# --- deletes and counts ---------------------------------------------------


def test_delete_manual_selects_by_filter_not_by_id(store):
    vector_store, client = store
    vector_store.delete_manual("m1")

    selector = client.delete.call_args.kwargs["points_selector"]
    assert selector.filter.must[0].match.value == "m1"
    # A delete is waited on: the caller commits SQLite in the same breath.
    assert client.delete.call_args.kwargs["wait"] is True


def test_count_is_exact(store):
    vector_store, client = store
    client.count.return_value = MagicMock(count=42)
    assert vector_store.count() == 42
    assert client.count.call_args.kwargs["exact"] is True


# --- collection lifecycle -------------------------------------------------


def test_a_collection_is_built_with_the_measured_graph_parameters(store):
    from mcp_manual_walker.embeddings import HNSW_EF_CONSTRUCTION, HNSW_MAX_NEIGHBORS

    vector_store, client = store
    vector_store._ready = False
    client.collection_exists.return_value = False

    vector_store._ensure_collection(1024)

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["vectors_config"].size == 1024
    assert kwargs["vectors_config"].distance == models.Distance.COSINE
    assert kwargs["hnsw_config"].m == HNSW_MAX_NEIGHBORS
    assert kwargs["hnsw_config"].ef_construct == HNSW_EF_CONSTRUCTION


def test_the_filtered_fields_get_payload_indexes(store):
    """Without them Qdrant cannot use the filter while traversing the graph,
    and every search this application issues carries a manual_id."""
    vector_store, client = store
    vector_store._ready = False
    client.collection_exists.return_value = False

    vector_store._ensure_collection(1024)

    indexed = {
        call.kwargs["field_name"]
        for call in client.create_payload_index.call_args_list
    }
    assert indexed == {"manual_id", "bookmark_id"}


def test_an_existing_collection_is_left_alone(store):
    vector_store, client = store
    vector_store._ready = False
    client.collection_exists.return_value = True

    vector_store._ensure_collection(1024)

    client.create_collection.assert_not_called()


def test_the_dimension_can_come_from_the_first_batch(store):
    """An import can be asked to create a store with no usable embedding model;
    the vectors it is about to write state the dimension just as well."""
    vector_store, client = store
    vector_store._ready = False
    vector_store._dimension = None
    client.collection_exists.return_value = False

    vector_store.add([Chunk(id="c0", document="x", embedding=[0.0] * 768)])

    assert client.create_collection.call_args.kwargs["vectors_config"].size == 768


def test_creating_a_collection_from_vectorless_chunks_is_refused(store):
    vector_store, _ = store
    vector_store._ready = False
    vector_store._dimension = None

    with pytest.raises(ValueError, match="carry no vectors"):
        vector_store.add([Chunk(id="c0", document="x")])


def test_opening_a_missing_collection_says_how_to_make_one():
    client = MagicMock()
    client.collection_exists.return_value = False

    with (
        patch.object(QdrantVectorStore, "_connect", return_value=client),
        patch.object(settings, "QDRANT_COLLECTION", COLLECTION),
    ):
        with pytest.raises(RuntimeError, match="db_manager build"):
            QdrantVectorStore.open()


# --- the model name, which Qdrant cannot hold ------------------------------


def test_the_model_name_is_recorded_when_a_store_is_created(sqlite_db):
    from mcp_manual_walker.vector_store import recorded_embedding_model

    client = MagicMock()
    client.collection_exists.return_value = False
    embedder = MagicMock(dimension=1024, model_name="Qwen/Qwen3-Embedding-0.6B")

    with (
        patch.object(QdrantVectorStore, "_connect", return_value=client),
        patch.object(settings, "QDRANT_COLLECTION", COLLECTION),
    ):
        store = QdrantVectorStore.open(embedder=embedder, create=True)

    assert recorded_embedding_model() == "Qwen/Qwen3-Embedding-0.6B"
    assert store.embedding_model == "Qwen/Qwen3-Embedding-0.6B"


def test_an_existing_store_keeps_the_name_it_was_built_with(sqlite_db):
    """Otherwise opening it under a different EMBEDDING_MODEL would overwrite
    the very fact that is supposed to catch the mismatch."""
    from mcp_manual_walker.vector_store import record_embedding_model

    record_embedding_model("intfloat/multilingual-e5-small")

    client = MagicMock()
    client.collection_exists.return_value = True
    embedder = MagicMock(dimension=1024, model_name="Qwen/Qwen3-Embedding-0.6B")

    with (
        patch.object(QdrantVectorStore, "_connect", return_value=client),
        patch.object(settings, "QDRANT_COLLECTION", COLLECTION),
    ):
        store = QdrantVectorStore.open(embedder=embedder, create=True)

    assert store.embedding_model == "intfloat/multilingual-e5-small"


def test_a_reset_drops_the_collection_and_the_recorded_name(sqlite_db):
    from mcp_manual_walker.vector_store import record_embedding_model

    record_embedding_model("Qwen/Qwen3-Embedding-0.6B")
    client = MagicMock()
    client.collection_exists.return_value = True

    with (
        patch.object(QdrantVectorStore, "_connect", return_value=client),
        patch.object(settings, "QDRANT_COLLECTION", COLLECTION),
    ):
        QdrantVectorStore.reset()

    client.delete_collection.assert_called_once_with(COLLECTION)
    from mcp_manual_walker.vector_store import recorded_embedding_model

    assert recorded_embedding_model() is None


# --- against a real server ------------------------------------------------

QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL")
live = pytest.mark.skipif(
    not QDRANT_TEST_URL,
    reason="set QDRANT_TEST_URL to run against a real Qdrant server",
)


@pytest.fixture
def live_store(sqlite_db):
    """A real, empty collection, dropped again afterwards."""
    name = f"mmw_test_{uuid.uuid4().hex[:8]}"
    embedder = MagicMock(dimension=8, model_name="test/model")
    with (
        patch.object(settings, "QDRANT_URL", QDRANT_TEST_URL),
        patch.object(settings, "QDRANT_COLLECTION", name),
        patch.object(settings, "QDRANT_API_KEY", ""),
    ):
        store = QdrantVectorStore.open(embedder=embedder, create=True)
        try:
            yield store
        finally:
            with patch.object(settings, "QDRANT_COLLECTION", name):
                QdrantVectorStore.reset()
            store.close()


def sample_chunks():
    """Two manuals, three sections, one figure -- the shapes the app stores."""
    return [
        Chunk(id="m1_0", document="alpha text", embedding=[1.0] + [0.0] * 7,
              metadata={"manual_id": "m1", "bookmark_id": "b1", "type": "text",
                        "chunk_index": 0.0, "source": "one.pdf"}),
        Chunk(id="m1_1", document="beta table", embedding=[0.0, 1.0] + [0.0] * 6,
              metadata={"manual_id": "m1", "bookmark_id": "b2", "type": "table",
                        "chunk_index": 1.0, "source": "one.pdf", "page": 4}),
        Chunk(id="m1_2", document="gamma figure", embedding=[0.0, 0.0, 1.0] + [0.0] * 5,
              metadata={"manual_id": "m1", "bookmark_id": "b2", "type": "figure",
                        "chunk_index": 2.0, "source": "one.pdf", "page": 5,
                        "figure_id": "fig-1"}),
        Chunk(id="m2_0", document="delta text", embedding=[0.0] * 3 + [1.0, 0.0] * 2 +
              [0.0], metadata={"manual_id": "m2", "bookmark_id": "b9", "type": "text",
                               "chunk_index": 0.0, "source": "two.pdf"}),
    ]


@live
def test_live_a_chunk_survives_the_round_trip_unchanged(live_store):
    """The export archive is written straight out of what scroll() returns."""
    original = sample_chunks()[1]
    live_store.add([original])

    read_back = next(live_store.scroll(ChunkFilter(manual_ids=["m1"]),
                                       with_embedding=True))

    assert read_back.id == original.id
    assert read_back.document == original.document
    assert read_back.metadata == original.metadata
    assert read_back.embedding == pytest.approx(list(original.embedding), abs=1e-6)


@live
def test_live_the_filters_actually_select(live_store):
    live_store.add(sample_chunks())

    assert live_store.count() == 4

    by_manual = {c.id for c in live_store.scroll(ChunkFilter(manual_ids=["m1"]))}
    assert by_manual == {"m1_0", "m1_1", "m1_2"}

    by_section = {
        c.id for c in live_store.scroll(
            ChunkFilter(manual_ids=["m1"], bookmark_ids=["b2"])
        )
    }
    assert by_section == {"m1_1", "m1_2"}

    both_sections = {
        c.id for c in live_store.scroll(
            ChunkFilter(manual_ids=["m1"], bookmark_ids=["b1", "b2"])
        )
    }
    assert both_sections == {"m1_0", "m1_1", "m1_2"}


@live
def test_live_a_re_import_overwrites_rather_than_duplicates(live_store):
    """This is what the deterministic point id buys."""
    chunks = sample_chunks()
    live_store.add(chunks)
    live_store.add(chunks)

    assert live_store.count() == 4


@live
def test_live_search_ranks_by_similarity_and_respects_the_filter(live_store):
    live_store.add(sample_chunks())

    hits = live_store.search([1.0] + [0.0] * 7, limit=3)
    assert hits[0].id == "m1_0"
    assert hits[0].score > hits[-1].score

    # The shape search_manual actually issues.
    scoped = live_store.search(
        [0.0, 1.0] + [0.0] * 6, limit=5, where=ChunkFilter(manual_ids=["m1"])
    )
    assert {h.id for h in scoped} == {"m1_0", "m1_1", "m1_2"}
    assert scoped[0].id == "m1_1"

    other = live_store.search([1.0] + [0.0] * 7, limit=5,
                              where=ChunkFilter(manual_ids=["m2"]))
    assert {h.id for h in other} == {"m2_0"}


@live
def test_live_get_returns_the_ids_asked_for_in_that_order(live_store):
    live_store.add(sample_chunks())

    got = live_store.get(["m1_2", "m1_0", "absent"])

    assert [c.id for c in got] == ["m1_2", "m1_0"]
    assert got[0].metadata["figure_id"] == "fig-1"


@live
def test_live_deleting_a_manual_leaves_the_others(live_store):
    live_store.add(sample_chunks())

    live_store.delete_manual("m1")

    assert live_store.count() == 1
    assert {c.id for c in live_store.scroll(ChunkFilter())} == {"m2_0"}


@live
def test_live_scrolling_the_whole_corpus_pages_through_everything(live_store):
    many = [
        Chunk(id=f"big_{i}", document=f"chunk {i}", embedding=[float(i % 8 == j)
              for j in range(8)] or [1.0] + [0.0] * 7,
              metadata={"manual_id": "big", "chunk_index": float(i)})
        for i in range(250)
    ]
    live_store.add(many)

    seen = [c.id for c in live_store.scroll(ChunkFilter(), batch_size=64)]

    assert len(seen) == 250
    assert len(set(seen)) == 250
