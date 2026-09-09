"""Tests for the OpenAI-compatible embedding backend.

The remote backend has to be interchangeable with the local one, and the ways
it can fail to be are quiet ones: a missing instruction prefix, an unnormalised
vector or a reordered response all produce perfectly well-formed numbers that
simply rank the wrong chunks first. Each of those is pinned here.
"""

from unittest.mock import patch

import pytest

from mcp_manual_walker.config import settings
from mcp_manual_walker.embeddings import (
    _KNOWN_PROMPTS,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)

QWEN_QUERY_PREFIX = _KNOWN_PROMPTS["Qwen/Qwen3-Embedding-0.6B"]["query"]

API_BASE = "http://embed.invalid:13305/v1"


def make_embedder(**overrides) -> OpenAICompatibleEmbedder:
    kwargs = {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "api_base": API_BASE,
        "api_model": "Qwen3-Embedding-0.6B-GGUF",
        "api_key": "",
        "query_prefix": None,
        "document_prefix": None,
        "batch_size": 2,
        "timeout": 5.0,
        "max_retries": 3,
    }
    kwargs.update(overrides)
    return OpenAICompatibleEmbedder(**kwargs)


class FakeResponse:
    def __init__(self, vectors, start_index=0, shuffle=False):
        data = [
            {"index": start_index + i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ]
        self._body = {"data": list(reversed(data)) if shuffle else data}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_documents_get_no_prefix_and_queries_get_the_instruction():
    """Qwen3-Embedding instructs queries and leaves passages bare.

    The endpoint cannot tell us this, so the backend carries the prompts.
    """
    embedder = make_embedder()

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_documents(["a passage"])
    assert post.call_args.kwargs["json"]["input"] == ["a passage"]

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_query("how do I reset it")
    assert post.call_args.kwargs["json"]["input"] == [
        QWEN_QUERY_PREFIX + "how do I reset it"
    ]


def test_an_explicit_prefix_overrides_the_known_prompt():
    embedder = make_embedder(query_prefix="query: ", document_prefix="passage: ")

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_query("x")
    assert post.call_args.kwargs["json"]["input"] == ["query: x"]


def test_an_unknown_model_falls_back_to_no_prefix_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        embedder = make_embedder(model_name="some/unknown-embedder")
    assert embedder.query_prefix == ""
    assert "cannot read them from the model" in caplog.text


def test_the_recorded_model_name_is_not_the_endpoints_id():
    """The collection records a vector space, not a deployment detail.

    A GGUF of Qwen3-Embedding-0.6B behind llama.cpp produces vectors that are
    interchangeable with the local checkpoint's, so reporting the endpoint's id
    here would make check_collection_model reject a database it can read.
    """
    embedder = make_embedder()
    assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_documents(["a"])
    assert post.call_args.kwargs["json"]["model"] == "Qwen3-Embedding-0.6B-GGUF"


def test_inputs_are_split_into_batches():
    embedder = make_embedder(batch_size=2)

    with patch("httpx.post") as post:
        post.side_effect = [
            FakeResponse([[1.0, 0.0], [0.0, 1.0]]),
            FakeResponse([[1.0, 0.0]], start_index=0),
        ]
        vectors = embedder.embed_documents(["a", "b", "c"])

    assert post.call_count == 2
    assert [call.kwargs["json"]["input"] for call in post.call_args_list] == [
        ["a", "b"],
        ["c"],
    ]
    assert len(vectors) == 3


def test_a_reordered_response_is_put_back_in_request_order():
    """The API does not promise the results come back in the order sent."""
    embedder = make_embedder(batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0], [0.0, 1.0]], shuffle=True)
        vectors = embedder.embed_documents(["first", "second"])

    assert vectors[0] == pytest.approx([1.0, 0.0])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_vectors_are_normalized():
    """The collections use cosine; an endpoint is not obliged to normalise."""
    embedder = make_embedder(batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[3.0, 4.0]])
        vector = embedder.embed_query("x")

    assert vector == pytest.approx([0.6, 0.8])


def test_a_short_response_is_an_error():
    """Silently returning fewer vectors than inputs would misalign every chunk."""
    embedder = make_embedder(batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
            embedder.embed_documents(["a", "b"])


def test_a_failed_request_is_retried_then_succeeds():
    embedder = make_embedder(batch_size=8, max_retries=3)

    with patch("httpx.post") as post, patch("time.sleep"):
        post.side_effect = [
            RuntimeError("connection reset"),
            FakeResponse([[1.0, 0.0]]),
        ]
        vector = embedder.embed_query("x")

    assert post.call_count == 2
    assert vector == pytest.approx([1.0, 0.0])


def test_a_permanently_failed_request_raises():
    embedder = make_embedder(batch_size=8, max_retries=2)

    with patch("httpx.post") as post, patch("time.sleep"):
        post.side_effect = RuntimeError("connection reset")
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            embedder.embed_query("x")


def test_an_api_key_is_sent_as_a_bearer_token():
    embedder = make_embedder(api_key="secret", batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_query("x")

    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"


def test_no_api_key_sends_no_authorization_header():
    embedder = make_embedder(batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_query("x")

    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_the_dimension_is_probed_once():
    embedder = make_embedder(batch_size=8)

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[0.0] * 1024])
        assert embedder.dimension == 1024
        assert embedder.dimension == 1024

    assert post.call_count == 1


def test_on_device_is_a_no_op():
    """The model is somebody else's process, so there is nothing to move."""
    embedder = make_embedder()
    with embedder.on_device():
        pass


def test_get_embedder_selects_the_remote_backend():
    with (
        patch.object(settings, "EMBEDDING_BACKEND", "openai"),
        patch.object(settings, "EMBEDDING_API_BASE", API_BASE),
        patch.object(settings, "EMBEDDING_API_MODEL", "Qwen3-Embedding-0.6B-GGUF"),
    ):
        embedder = get_embedder()

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_name == settings.EMBEDDING_MODEL


def test_get_embedder_refuses_the_remote_backend_without_a_base_url(caplog):
    with (
        patch.object(settings, "EMBEDDING_BACKEND", "openai"),
        patch.object(settings, "EMBEDDING_API_BASE", ""),
        caplog.at_level("ERROR"),
    ):
        assert get_embedder() is None
    assert "EMBEDDING_API_BASE is empty" in caplog.text


def test_get_embedder_rejects_an_unknown_backend(caplog):
    with (
        patch.object(settings, "EMBEDDING_BACKEND", "pinecone"),
        caplog.at_level("ERROR"),
    ):
        assert get_embedder() is None
    assert "expected 'local' or 'openai'" in caplog.text


def test_the_url_is_built_from_the_base():
    embedder = make_embedder(api_base=API_BASE + "/")

    with patch("httpx.post") as post:
        post.return_value = FakeResponse([[1.0, 0.0]])
        embedder.embed_query("x")

    assert post.call_args.args[0] == f"{API_BASE}/embeddings"


def test_embedding_nothing_makes_no_request():
    embedder = make_embedder()
    with patch("httpx.post") as post:
        assert embedder.embed_documents([]) == []
    post.assert_not_called()


def test_the_protocol_shape_matches_the_local_backend():
    """Both backends are used through the same attribute names.

    Checked on the class, not on an instance: `dimension` is a property that
    probes the endpoint, and hasattr() would evaluate it.
    """
    for name in ("model_name", "dimension", "query_prefix", "document_prefix"):
        assert isinstance(
            getattr(OpenAICompatibleEmbedder, name), property
        ), f"{name} is not a property"
        assert isinstance(getattr(SentenceTransformerEmbedder, name), property)
    for name in ("embed_documents", "embed_query", "on_device", "__call__"):
        assert callable(getattr(OpenAICompatibleEmbedder, name))
        assert callable(getattr(SentenceTransformerEmbedder, name))
