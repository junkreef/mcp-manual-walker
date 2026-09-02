import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcp_manual_walker.config import settings
from mcp_manual_walker.embeddings import (
    EMBEDDING_MODEL_METADATA_KEY,
    SentenceTransformerEmbedder,
    _resolve_device,
    check_collection_model,
    get_embedder,
)


@pytest.fixture
def mock_sentence_transformers():
    """
    Installs a fake sentence_transformers module.

    The real package (and torch) are not installed in the test environment, and
    loading Qwen3-Embedding would require a model download anyway.
    """
    mock_module = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_module.SentenceTransformer.return_value = mock_model

    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        yield SimpleNamespace(module=mock_module, model=mock_model)


def test_constructor_configures_the_model(mock_sentence_transformers):
    """The model must be loaded left-padded, on the requested device."""
    with (
        patch.object(settings, "EMBEDDING_DEVICE", "cpu"),
        patch.object(settings, "EMBEDDING_MAX_SEQ_LENGTH", 512),
    ):
        embedder = get_embedder()

    mock_sentence_transformers.module.SentenceTransformer.assert_called_once_with(
        settings.EMBEDDING_MODEL,
        device="cpu",
        tokenizer_kwargs={"padding_side": "left"},
    )
    assert mock_sentence_transformers.model.max_seq_length == 512
    assert embedder.model_name == settings.EMBEDDING_MODEL
    assert embedder.dimension == 1024


def test_embed_documents(mock_sentence_transformers):
    """Documents are embedded without an instruction prefix."""
    mock_sentence_transformers.model.encode.return_value = np.array(
        [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
    )

    embedder = SentenceTransformerEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        device="cpu",
        query_prefix=settings.EMBEDDING_QUERY_PREFIX,
        document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
        max_seq_length=512,
        batch_size=8,
    )
    result = embedder.embed_documents(["doc a", "doc b"])

    mock_sentence_transformers.model.encode.assert_called_once_with(
        ["doc a", "doc b"],
        prompt=settings.EMBEDDING_DOCUMENT_PREFIX or None,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert result[0] == pytest.approx([0.1, 0.2])
    assert result[1] == pytest.approx([0.3, 0.4])


def test_call_is_an_alias_of_embed_documents(mock_sentence_transformers):
    """The Chroma-style callable must behave like embed_documents."""
    embedder = SentenceTransformerEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        device="cpu",
        query_prefix=settings.EMBEDDING_QUERY_PREFIX,
        document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
        max_seq_length=512,
        batch_size=8,
    )
    result = embedder(["doc a"])

    kwargs = mock_sentence_transformers.model.encode.call_args.kwargs
    assert kwargs["prompt"] == (settings.EMBEDDING_DOCUMENT_PREFIX or None)
    assert result[0] == pytest.approx([0.1, 0.2, 0.3])


def test_embed_query_uses_the_instruction_prefix(mock_sentence_transformers):
    """Queries carry the Qwen3 instruction prefix and return a flat vector."""
    embedder = SentenceTransformerEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        device="cpu",
        query_prefix=settings.EMBEDDING_QUERY_PREFIX,
        document_prefix=settings.EMBEDDING_DOCUMENT_PREFIX,
        max_seq_length=512,
        batch_size=8,
    )
    result = embedder.embed_query("how to reset")

    mock_sentence_transformers.model.encode.assert_called_once_with(
        ["how to reset"],
        prompt=settings.EMBEDDING_QUERY_PREFIX,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert result == pytest.approx([0.1, 0.2, 0.3])


def test_get_embedder_without_sentence_transformers():
    """Without the library installed the factory must return None, not raise."""
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        assert get_embedder() is None


def test_resolve_device_explicit():
    """An explicit device string must be passed through untouched."""
    assert _resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_without_torch():
    """Without torch available, "auto" must fall back to CPU."""
    with patch.dict(sys.modules, {"torch": None}):
        assert _resolve_device("auto") == "cpu"


def test_check_collection_model_matching():
    """A collection built with the expected model passes."""
    collection = SimpleNamespace(
        metadata={EMBEDDING_MODEL_METADATA_KEY: settings.EMBEDDING_MODEL}
    )
    check_collection_model(collection, settings.EMBEDDING_MODEL)


def test_check_collection_model_mismatch():
    """A collection built with another model must be rejected."""
    collection = SimpleNamespace(
        metadata={EMBEDDING_MODEL_METADATA_KEY: "intfloat/multilingual-e5-small"}
    )
    with pytest.raises(RuntimeError) as excinfo:
        check_collection_model(collection, settings.EMBEDDING_MODEL)

    message = str(excinfo.value)
    assert "intfloat/multilingual-e5-small" in message
    assert settings.EMBEDDING_MODEL in message
    assert "--reset" in message


def test_check_collection_model_missing_key():
    """A legacy collection without the metadata key must be rejected too."""
    collection = SimpleNamespace(metadata={"description": "Chunks from PDF manuals"})
    with pytest.raises(RuntimeError) as excinfo:
        check_collection_model(collection, settings.EMBEDDING_MODEL)

    assert "predates" in str(excinfo.value)
