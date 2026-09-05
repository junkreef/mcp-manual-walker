"""What `db_manager` drags in just to start.

The builder pulls in Docling, torch, transformers and PIL: 936 MB resident.
Every subcommand was paying it at import time, including `watch`, which reads
a text file and draws it -- and which runs alongside a build that is already
close to the host's memory ceiling.
"""

import subprocess
import sys
import textwrap

HEAVY = ("torch", "docling", "chromadb", "transformers", "sentence_transformers")


def modules_after(statement):
    """Heavy modules resident after running `statement` in a fresh process."""
    code = textwrap.dedent(f"""
        import sys
        {statement}
        heavy = [m for m in {HEAVY!r}
                 if any(k == m or k.startswith(m + ".") for k in sys.modules)]
        print(",".join(sorted(heavy)))
    """)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(filter(None, out.stdout.strip().split(",")))


def test_importing_the_cli_pulls_in_nothing_heavy():
    assert modules_after("import mcp_manual_walker.db_manager") == set()


def test_the_monitor_pulls_in_nothing_heavy():
    # `watch` is the case that matters: it reads a JSONL file and renders it.
    assert modules_after("from mcp_manual_walker.tui import watch") == set()


def test_the_progress_reader_pulls_in_nothing_heavy():
    assert modules_after("from mcp_manual_walker.progress import read_progress") == set()


def test_the_builder_is_reachable_when_a_build_actually_runs():
    # Laziness must not mean unreachable: command_build imports it on the way in.
    from mcp_manual_walker import db_manager

    assert "from mcp_manual_walker.builder import build" in __import__(
        "inspect"
    ).getsource(db_manager.command_build)


def test_chroma_is_loaded_on_demand():
    from mcp_manual_walker import db_manager

    assert db_manager._load_chromadb() is not None
    assert db_manager.chromadb is not None
