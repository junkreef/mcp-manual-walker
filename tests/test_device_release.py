"""Giving the GPU back between documents.

Two things keep a GPU occupied by an idle embedder: torch's caching allocator
holds the activation peak in its own pool, and the weights stay resident.
Measured in the builder's parent at 5114 MB long after its last batch against
1346 MB before its first. On a device shared with the Docling workers that is
a reservation, not a cache -- the worker that takes the freed slot finds the
GPU still full.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_manual_walker.embeddings import SentenceTransformerEmbedder


def embedder(device):
    """A bare embedder: on_device touches only the model and the allocator."""
    e = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    e._device = device
    e._is_accelerated = not str(device).startswith("cpu")
    e.model = SimpleNamespace(to=MagicMock())
    e._empty_cache = MagicMock()
    return e


def moves(e):
    return [call.args[0] for call in e.model.to.call_args_list]


def test_a_cpu_embedder_moves_nothing():
    e = embedder("cpu")
    with e.on_device():
        pass
    assert moves(e) == []
    e._empty_cache.assert_not_called()


def test_the_model_goes_to_the_device_and_comes_back():
    e = embedder("cuda")
    with e.on_device():
        assert moves(e) == ["cuda"]
    assert moves(e) == ["cuda", "cpu"]


def test_the_allocator_pool_is_emptied_on_the_way_out():
    # Moving the weights off is not enough on its own: the activation blocks
    # torch has cached are the larger half.
    e = embedder("cuda")
    with e.on_device():
        e._empty_cache.assert_not_called()
    e._empty_cache.assert_called_once()


def test_the_device_is_given_back_even_when_the_block_raises():
    e = embedder("cuda")
    with pytest.raises(RuntimeError):
        with e.on_device():
            raise RuntimeError("embedding blew up")
    assert moves(e) == ["cuda", "cpu"]
    e._empty_cache.assert_called_once()


def test_a_failure_to_move_does_not_fail_the_build():
    # A document embedded on the wrong device is still a document; a build
    # that dies because a .to() failed is not.
    e = embedder("cuda")
    e.model.to.side_effect = RuntimeError("device fell over")
    with e.on_device():
        pass
    e._empty_cache.assert_called_once()


def test_a_specific_device_index_is_honoured():
    e = embedder("cuda:1")
    with e.on_device():
        pass
    assert moves(e) == ["cuda:1", "cpu"]


def test_repeated_use_returns_the_device_every_time():
    e = embedder("cuda")
    for _ in range(3):
        with e.on_device():
            pass
    assert moves(e) == ["cuda", "cpu"] * 3
    assert e._empty_cache.call_count == 3
