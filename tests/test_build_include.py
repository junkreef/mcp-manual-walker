"""Subset selection for `db_manager build --include`."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_manual_walker.builder import select_pdf_files
from mcp_manual_walker.db_manager import command_build

CORPUS = [
    "zOS/V3R1/bpxbd00.pdf",
    "zOS/V3R1/nested/deep.pdf",
    "zOS/V3R2/bpxbd00.pdf",
    "Db2 for zOS/v13/admin.pdf",
    "loose.pdf",
    "zOS/V3R1/notes.txt",
]


@pytest.fixture
def corpus(tmp_path):
    for rel in CORPUS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n")
    return tmp_path


def rels(paths, root):
    return sorted(p.relative_to(root).as_posix() for p in paths)


def test_no_patterns_takes_every_pdf(corpus):
    assert rels(select_pdf_files(corpus), corpus) == [
        "Db2 for zOS/v13/admin.pdf",
        "loose.pdf",
        "zOS/V3R1/bpxbd00.pdf",
        "zOS/V3R1/nested/deep.pdf",
        "zOS/V3R2/bpxbd00.pdf",
    ]


def test_empty_pattern_list_is_the_same_as_none(corpus):
    assert select_pdf_files(corpus, []) == select_pdf_files(corpus, None)


def test_directory_pattern_reaches_below_the_directory(corpus):
    # '*' spans '/' in fnmatch, which is what makes a single pattern enough to
    # select a product directory and everything nested under it.
    assert rels(select_pdf_files(corpus, ["zOS/V3R1/*"]), corpus) == [
        "zOS/V3R1/bpxbd00.pdf",
        "zOS/V3R1/nested/deep.pdf",
    ]


def test_patterns_are_anchored_at_the_root_not_the_basename(corpus):
    # The same file name exists under V3R1 and V3R2; the prefix has to separate
    # them, or a subset build would silently pull in the other release.
    selected = select_pdf_files(corpus, ["zOS/V3R2/*"])
    assert rels(selected, corpus) == ["zOS/V3R2/bpxbd00.pdf"]


def test_several_patterns_union(corpus):
    selected = select_pdf_files(corpus, ["zOS/V3R2/*", "loose.pdf"])
    assert rels(selected, corpus) == ["loose.pdf", "zOS/V3R2/bpxbd00.pdf"]


def test_a_file_matching_two_patterns_is_listed_once(corpus):
    selected = select_pdf_files(corpus, ["zOS/*", "*/V3R1/*"])
    assert len(selected) == len(set(selected))
    assert rels(selected, corpus) == [
        "zOS/V3R1/bpxbd00.pdf",
        "zOS/V3R1/nested/deep.pdf",
        "zOS/V3R2/bpxbd00.pdf",
    ]


def test_matching_is_case_sensitive(corpus):
    assert select_pdf_files(corpus, ["zos/v3r1/*"]) == []


def test_patterns_never_widen_the_selection_to_non_pdfs(corpus):
    assert select_pdf_files(corpus, ["zOS/V3R1/*.txt"]) == []


def test_no_match_returns_empty_rather_than_everything(corpus):
    assert select_pdf_files(corpus, ["CICS/*"]) == []


def test_selection_is_deterministic(corpus):
    assert select_pdf_files(corpus, ["zOS/*"]) == select_pdf_files(corpus, ["zOS/*"])


def test_relative_paths_stay_anchored_at_pdf_dir(corpus):
    # The point of --include: narrowing the scan must not move the anchor that
    # every stored relative_path is derived from.
    (selected,) = select_pdf_files(corpus, ["loose.pdf"])
    assert selected.relative_to(corpus).as_posix() == "loose.pdf"


# `build` is imported inside command_build (the builder costs 936 MB at
# import time), so it is patched where it is defined rather than where it
# is used -- the builder is imported lazily.
def build_args(tmp_path, **overrides):
    args = dict(
        pdf_dir=str(tmp_path),
        reset=False,
        save_markdown=False,
        include=None,
        progress_file=str(tmp_path / "progress.jsonl"),
        no_progress=True,
        min_pages=None,
        max_pages=None,
    )
    args.update(overrides)
    return Namespace(**args)


def test_cli_forwards_include_to_build(tmp_path):
    args = build_args(tmp_path, include=["zOS/V3R1/*"])
    with patch("mcp_manual_walker.builder.build") as mock_build:
        command_build(args)
    mock_build.assert_called_once_with(
        Path(tmp_path), False, False, ["zOS/V3R1/*"], None,
        min_pages=None, max_pages=None,
    )


def test_cli_passes_none_when_the_flag_is_absent(tmp_path):
    args = build_args(tmp_path)
    with patch("mcp_manual_walker.builder.build") as mock_build:
        command_build(args)
    mock_build.assert_called_once_with(
        Path(tmp_path), False, False, None, None, min_pages=None, max_pages=None
    )


def test_cli_passes_the_progress_file_through(tmp_path):
    args = build_args(tmp_path, no_progress=False)
    with patch("mcp_manual_walker.builder.build") as mock_build:
        command_build(args)
    assert mock_build.call_args.args[4] == tmp_path / "progress.jsonl"
