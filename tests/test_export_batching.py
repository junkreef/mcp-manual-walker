"""Writing an archive's chunks without holding the corpus in memory.

Two limits meet here. Chroma turns a `$in` clause into SQLite parameters and
the limit is reached by the number of *matching chunks*, not the number of ids:
measured on this corpus, 10 ids matching 11,125 chunks was fine and 20 ids was
not, and a manual holds anywhere from 328 to several thousand chunks. And the
previous export built one JSON object out of every chunk, which the OOM killer
ended at 504,346 of them.
"""

import json

import numpy as np
import pytest

from mcp_manual_walker.db_manager import (
    _chunks_from_legacy,
    _read_chunks,
    _write_chunks,
)


class FakeCollection:
    """Refuses a query matching too many chunks, as Chroma does."""

    def __init__(self, per_manual=2, chunk_limit=11000, embeddings=True):
        self.per_manual = per_manual
        self.chunk_limit = chunk_limit
        self.embeddings = embeddings
        self.queries = []

    def get(self, where, include):
        clause = where["manual_id"]
        ids = clause["$in"] if isinstance(clause, dict) else [clause]
        self.queries.append(len(ids))
        if len(ids) * self.per_manual > self.chunk_limit:
            raise RuntimeError("too many SQL variables")
        out = {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
        for mid in ids:
            for i in range(self.per_manual):
                out["ids"].append(f"{mid}_{i}")
                out["metadatas"].append({"manual_id": mid})
                out["documents"].append(f"text {mid} {i}")
                out["embeddings"].append(np.array([0.1, 0.2], dtype=np.float32))
        if not self.embeddings:
            out["embeddings"] = None
        return out


def written(tmp_path, col, ids):
    path = tmp_path / "chunks.jsonl"
    count = _write_chunks(col, ids, path)
    return count, list(_read_chunks(path))


def test_a_corpus_sized_export_asks_for_one_manual_at_a_time(tmp_path):
    # The limit is on matching chunks, so the only size that always fits is one.
    col = FakeCollection(per_manual=5000, chunk_limit=11000)
    _write_chunks(col, [f"m{i}" for i in range(369)], tmp_path / "c.jsonl")
    assert set(col.queries) == {1}
    assert len(col.queries) == 369


def test_every_chunk_is_written_exactly_once(tmp_path):
    count, chunks = written(tmp_path, FakeCollection(per_manual=3),
                            [f"m{i}" for i in range(250)])
    assert count == 750
    assert len({c["id"] for c in chunks}) == 750


def test_a_chunk_keeps_its_four_fields_together(tmp_path):
    _, chunks = written(tmp_path, FakeCollection(), ["m0", "m1"])
    for c in chunks:
        assert c["metadata"]["manual_id"] == c["id"].rsplit("_", 1)[0]
        assert c["document"].startswith(f"text {c['metadata']['manual_id']}")
        assert isinstance(c["embedding"], list)


def test_embeddings_survive_the_round_trip_as_plain_lists(tmp_path):
    _, chunks = written(tmp_path, FakeCollection(), ["m0"])
    assert chunks[0]["embedding"] == pytest.approx([0.1, 0.2], abs=1e-6)


def test_one_line_per_chunk(tmp_path):
    path = tmp_path / "c.jsonl"
    _write_chunks(FakeCollection(per_manual=3), ["m0", "m1"], path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 6
    assert all(json.loads(line)["id"] for line in lines)


def test_a_chunk_without_an_embedding_is_kept_with_a_null(tmp_path):
    # Dropping it here would silently change what the archive contains; the
    # importer is where that decision belongs.
    _, chunks = written(tmp_path, FakeCollection(embeddings=False), ["m0"])
    assert [c["embedding"] for c in chunks] == [None, None]


def test_no_chunks_at_all_writes_an_empty_file(tmp_path):
    count, chunks = written(tmp_path, FakeCollection(per_manual=0), ["m0", "m1"])
    assert count == 0
    assert chunks == []


@pytest.mark.parametrize("n", [0, 1, 2, 100, 369])
def test_it_handles_any_number_of_manuals(tmp_path, n):
    count, chunks = written(tmp_path, FakeCollection(), [f"m{i}" for i in range(n)])
    assert count == n * 2 == len(chunks)


def test_a_version_2_archive_reads_as_the_same_shape():
    legacy = {
        "ids": ["c0", "c1"],
        "embeddings": [[0.1], [0.2]],
        "metadatas": [{"manual_id": "m"}, {"manual_id": "m"}],
        "documents": ["a", "b"],
    }
    assert list(_chunks_from_legacy(legacy)) == [
        {"id": "c0", "embedding": [0.1], "metadata": {"manual_id": "m"}, "document": "a"},
        {"id": "c1", "embedding": [0.2], "metadata": {"manual_id": "m"}, "document": "b"},
    ]


def test_a_version_2_archive_with_no_embeddings_still_reads():
    legacy = {
        "ids": ["c0"],
        "embeddings": None,
        "metadatas": [{"manual_id": "m"}],
        "documents": ["a"],
    }
    assert list(_chunks_from_legacy(legacy))[0]["embedding"] is None


def test_an_empty_version_2_archive_yields_nothing():
    assert list(_chunks_from_legacy({})) == []


def test_blank_lines_in_the_chunk_file_are_skipped(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert [c["id"] for c in _read_chunks(path)] == ["a", "b"]
