import sys
from unittest.mock import MagicMock, patch

import pytest

from mcp_manual_walker.embeddings import (
    FastEmbedEmbeddingFunction,
    get_embedding_function,
)


@pytest.fixture(autouse=True)
def mock_fastembed_module():
    """
    Globally mock fastembed to prevent PyO3 initialization errors and allow testing
    without the actual library backend.
    """
    mock_fe = MagicMock()
    # Mock TextEmbedding class
    mock_fe.TextEmbedding = MagicMock()

    with patch.dict(sys.modules, {"fastembed": mock_fe}):
        yield mock_fe


def test_get_embedding_function_fastembed_installed():
    """
    Test that FastEmbedEmbeddingFunction is returned when fastembed is installed
    and sentence-transformers is NOT.
    """
    # Simulate sentence_transformers missing
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        ef = get_embedding_function()

        assert isinstance(ef, FastEmbedEmbeddingFunction)
        assert ef.model is not None


def test_get_embedding_function_sentence_transformers_installed():
    """
    Test that SentenceTransformerEmbeddingFunction is returned when sentence-transformers IS installed.
    """
    mock_st = MagicMock()

    # We need to ensure chromadb.utils.embedding_functions uses our SentenceTransformerEmbeddingFunction
    # Since we are running in an environment where chromadb IS installed (via dev deps),
    # we can use standard patch on the real module path.

    with patch(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    ) as mock_st_ef_cls:
        # Simulate sentence_transformers installed
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            # We also need to make sure the import inside get_embedding_function succeeds.
            # importing sentence_transformers will succeed due to sys.modules patch.

            ef = get_embedding_function()

            mock_st_ef_cls.assert_called_with(
                model_name="intfloat/multilingual-e5-small"
            )
            assert ef == mock_st_ef_cls.return_value


def test_fastembed_embedding_function_call(mock_fastembed_module):
    """
    Test that FastEmbedEmbeddingFunction calls the model.embed method correctly.
    """
    # Configure the mock set up by the fixture
    mock_model_instance = MagicMock()
    mock_fastembed_module.TextEmbedding.return_value = mock_model_instance
    mock_model_instance.embed.return_value = [[0.1, 0.2, 0.3]]

    ef = FastEmbedEmbeddingFunction()
    result = ef(["test document"])

    mock_fastembed_module.TextEmbedding.assert_called()
    mock_model_instance.embed.assert_called_with(["test document"])

    # helper to convert numpy arrays to lists if needed
    result_list = [x.tolist() if hasattr(x, "tolist") else x for x in result]
    assert len(result_list) == 1
    assert result_list[0] == pytest.approx([0.1, 0.2, 0.3])
