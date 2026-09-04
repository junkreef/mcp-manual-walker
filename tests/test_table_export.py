"""Exporting tables to markdown without paying for the document each time.

`TableItem.export_to_markdown(doc)` builds a MarkdownDocSerializer per call,
and constructing one revalidates the whole document -- pydantic clamps every
table cell's bounding box on every page. The cost is (tables x cells), which
on a table-dense manual dominated the entire build.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from mcp_manual_walker.chunking import _make_table_exporter, _table_markdown


def test_one_serializer_serves_every_table(monkeypatch):
    built = []

    class Serializer:
        def __init__(self, doc):
            built.append(doc)

        def serialize(self, item):
            return SimpleNamespace(text=f"md:{item}")

    monkeypatch.setattr(
        "docling_core.transforms.serializer.markdown.MarkdownDocSerializer",
        Serializer,
    )
    export = _make_table_exporter("DOC")
    assert [export(f"t{i}") for i in range(50)] == [f"md:t{i}" for i in range(50)]
    # Fifty tables, one construction. That ratio is the whole point.
    assert len(built) == 1


def test_the_exported_text_is_stripped(monkeypatch):
    monkeypatch.setattr(
        "docling_core.transforms.serializer.markdown.MarkdownDocSerializer",
        lambda doc: SimpleNamespace(
            serialize=lambda item: SimpleNamespace(text="  | a |\n  ")
        ),
    )
    assert _make_table_exporter("DOC")("t") == "| a |"


def test_a_serializer_that_cannot_be_built_falls_back(monkeypatch, caplog):
    # A docling_core that moves the serializer should cost speed, not the build.
    def boom(doc):
        raise ImportError("moved in 3.0")

    monkeypatch.setattr(
        "docling_core.transforms.serializer.markdown.MarkdownDocSerializer", boom
    )
    item = SimpleNamespace(export_to_markdown=lambda doc: "fallback md")
    with caplog.at_level("WARNING", logger="mcp_manual_walker.chunking"):
        export = _make_table_exporter("DOC")
    assert export(item) == "fallback md"
    assert "quadratic" in caplog.text


def test_one_bad_table_falls_back_without_losing_the_rest(monkeypatch):
    class Serializer:
        def __init__(self, doc):
            pass

        def serialize(self, item):
            if item.bad:
                raise RuntimeError("this table is malformed")
            return SimpleNamespace(text="good")

    monkeypatch.setattr(
        "docling_core.transforms.serializer.markdown.MarkdownDocSerializer",
        Serializer,
    )
    export = _make_table_exporter("DOC")
    ok = SimpleNamespace(bad=False, export_to_markdown=lambda doc: "unused")
    bad = SimpleNamespace(bad=True, export_to_markdown=lambda doc: "per-call md")
    assert export(ok) == "good"
    assert export(bad) == "per-call md"


def test_the_per_call_path_still_works_on_its_own():
    item = SimpleNamespace(export_to_markdown=lambda doc: "  md  ")
    assert _table_markdown(item, "DOC") == "md"


def test_an_item_that_cannot_export_yields_nothing():
    assert _table_markdown(SimpleNamespace(), "DOC") == ""


def test_a_deprecated_signature_without_doc_is_tolerated():
    call = MagicMock(side_effect=[TypeError("no doc argument"), "md"])
    assert _table_markdown(SimpleNamespace(export_to_markdown=call), "DOC") == "md"
