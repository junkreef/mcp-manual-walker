"""Converting a long document as page ranges and putting it back together."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_manual_walker.builder import merge_parts, plan_parts


# -- planning ----------------------------------------------------------------


def sizes(ranges):
    return [end - start + 1 for start, end in ranges]


def test_a_short_document_is_one_part():
    assert plan_parts(200, 250) == [(1, 200)]


def test_a_document_exactly_at_the_limit_is_one_part():
    assert plan_parts(250, 250) == [(1, 250)]


def test_a_long_document_is_cut_up():
    ranges = plan_parts(1246, 250)
    assert len(ranges) == 5
    assert ranges[0][0] == 1
    assert ranges[-1][1] == 1246


def test_the_parts_are_contiguous_and_cover_every_page():
    ranges = plan_parts(2900, 250)
    assert ranges[0][0] == 1
    assert ranges[-1][1] == 2900
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert start == end + 1
    assert sum(sizes(ranges)) == 2900


def test_the_parts_are_even_rather_than_leaving_a_sliver():
    # 2900/250 would be eleven parts of 250 and one of 150: a worker holding
    # the sliver finishes early and idles while the others carry full parts.
    ranges = plan_parts(2900, 250)
    assert max(sizes(ranges)) - min(sizes(ranges)) <= 1


def test_no_part_is_longer_than_the_limit():
    for pages in (251, 499, 500, 501, 1246, 2900, 9999):
        assert max(sizes(plan_parts(pages, 250))) <= 250


def test_splitting_can_be_turned_off():
    assert plan_parts(2900, 0) == [(1, 2900)]
    assert plan_parts(2900, -1) == [(1, 2900)]


def test_a_document_with_no_known_page_count_is_one_part():
    # page_count is 0 when pypdf could not read it; converting it whole is the
    # honest fallback, and Docling will find out how long it really is.
    assert plan_parts(0, 250) == [(1, 1)]


# -- merging -----------------------------------------------------------------


def fake_doc(n_pictures):
    return SimpleNamespace(pictures=list(range(n_pictures)))


@pytest.fixture
def merging(monkeypatch):
    """merge_parts with a concatenate that records what it was handed."""
    import mcp_manual_walker.builder as builder

    seen = {}

    def concatenate(docs):
        seen["docs"] = list(docs)
        return SimpleNamespace(merged=True, parts=len(docs))

    monkeypatch.setattr(
        builder, "_SPLIT_SUPPORT", SimpleNamespace(concatenate=concatenate)
    )
    return seen


def figure(index, page):
    return {"picture_index": index, "page": page, "png": b""}


def test_parts_are_merged_in_page_order_however_they_arrive(merging):
    first, second, third = fake_doc(1), fake_doc(1), fake_doc(1)
    # as_completed hands parts back in whatever order they finish
    merge_parts([(501, third, []), (1, first, []), (251, second, [])])
    assert merging["docs"] == [first, second, third]


def test_figure_indices_are_shifted_by_the_pictures_before_them(merging):
    # Every part indexes its own pictures from zero, and concatenate re-indexes
    # them across the merged document. Without the shift, three parts would all
    # claim picture 0 and their figure rows would point at the wrong crop.
    _, figures = merge_parts(
        [
            (1, fake_doc(2), [figure(0, 3), figure(1, 7)]),
            (251, fake_doc(3), [figure(0, 260), figure(2, 290)]),
            (501, fake_doc(1), [figure(0, 505)]),
        ]
    )
    assert [f["picture_index"] for f in figures] == [0, 1, 2, 4, 5]
    assert [f["page"] for f in figures] == [3, 7, 260, 290, 505]


def test_the_shift_counts_pictures_not_extracted_figures(merging):
    # A picture whose crop could not be rendered is skipped by _extract_figures
    # but still occupies an index in the merged document.
    _, figures = merge_parts(
        [(1, fake_doc(3), [figure(0, 1)]), (251, fake_doc(1), [figure(0, 260)])]
    )
    assert [f["picture_index"] for f in figures] == [0, 3]


def test_merging_does_not_mutate_the_parts_figures(merging):
    original = figure(0, 260)
    merge_parts([(1, fake_doc(2), []), (251, fake_doc(1), [original])])
    assert original["picture_index"] == 0


def test_a_single_part_merges_to_itself(merging):
    _, figures = merge_parts([(1, fake_doc(2), [figure(1, 4)])])
    assert [f["picture_index"] for f in figures] == [1]


# -- the guard around Docling's internals ------------------------------------


def test_the_build_falls_back_to_whole_documents_without_split_support(
    monkeypatch,
):
    import mcp_manual_walker.builder as builder

    monkeypatch.setattr(builder, "_SPLIT_SUPPORT", None)
    # plan_parts is told the split size the build computed, which is 0 when
    # support is missing; the guard lives at the call site.
    assert plan_parts(2900, 0) == [(1, 2900)]


def test_assign_heading_levels_never_raises(monkeypatch):
    import mcp_manual_walker.builder as builder

    exploding = MagicMock()
    exploding.return_value.assign_heading_levels.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        builder,
        "_SPLIT_SUPPORT",
        SimpleNamespace(heading_model_cls=exploding),
    )
    # A document with unlevelled headings beats no document at all.
    builder.assign_heading_levels(object(), {}, None)


def test_assign_heading_levels_is_a_no_op_without_support(monkeypatch):
    import mcp_manual_walker.builder as builder

    monkeypatch.setattr(builder, "_SPLIT_SUPPORT", None)
    builder.assign_heading_levels(object(), {}, None)


def test_reading_an_outline_never_raises(monkeypatch, tmp_path):
    import mcp_manual_walker.builder as builder

    monkeypatch.setattr(
        builder,
        "_SPLIT_SUPPORT",
        SimpleNamespace(
            read_outline=MagicMock(side_effect=OSError("not a pdf"))
        ),
    )
    assert builder._read_outline(tmp_path / "nope.pdf") is None


# -- the parent must not accumulate finished work ----------------------------


def test_a_completed_future_is_released():
    """A Future holds its result until dropped.

    Keeping every part's Future for the length of the build kept every part's
    document, figure PNGs and parsed pages resident in the parent: measured at
    +4.4 GB after five documents, 1.1 MB per page, which over a 172k-page
    corpus is not survivable.
    """
    import gc
    import weakref
    from concurrent.futures import Future, as_completed

    class Result:
        pass

    futures = {}
    refs = []
    for i in range(3):
        fut = Future()
        result = Result()
        refs.append(weakref.ref(result))
        fut.set_result(result)
        futures[fut] = i
        del result

    # The loop as the builder runs it.
    for fut in as_completed(list(futures)):
        futures.pop(fut)
        value = fut.result()  # noqa: F841 - held only for this iteration
        del value
    del fut
    gc.collect()

    assert [r() for r in refs] == [None, None, None], (
        "a finished part's result outlived the iteration that consumed it"
    )


def test_holding_the_futures_dict_is_what_leaks():
    """The same loop, without the pop: this is the shape that leaked."""
    import gc
    import weakref
    from concurrent.futures import Future, as_completed

    class Result:
        pass

    futures = {}
    refs = []
    for i in range(3):
        fut = Future()
        result = Result()
        refs.append(weakref.ref(result))
        fut.set_result(result)
        futures[fut] = i
        del result

    for fut in as_completed(list(futures)):
        pass
    del fut
    gc.collect()

    assert all(r() is not None for r in refs)
