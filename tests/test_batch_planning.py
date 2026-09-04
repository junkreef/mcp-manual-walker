"""Grouping texts into batches under a token budget.

A batch is padded to its longest member, so its cost is len(batch) x longest.
Batching by row count alone prices every batch at its worst member.
"""

import pytest

from mcp_manual_walker.embeddings import SentenceTransformerEmbedder


def planner(budget, max_rows=32):
    """A bare embedder: plan_batches touches no model state."""
    planner = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    planner._token_budget = budget
    planner._batch_size = max_rows
    return planner


def cost(batch, lengths):
    """What the batch actually costs: padded to its longest member."""
    return len(batch) * max(lengths[i] for i in batch)


def test_every_text_lands_in_exactly_one_batch():
    lengths = [10, 500, 30, 4000, 20, 700]
    batches = planner(2000).plan_batches(lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_no_batch_exceeds_the_budget():
    lengths = [345] * 40 + [1313, 3836, 80, 22]
    budget = 4096
    batches = planner(budget).plan_batches(lengths)
    for batch in batches:
        assert len(batch) == 1 or cost(batch, lengths) <= budget


def test_short_texts_batch_widely_and_long_ones_narrowly():
    lengths = [100] * 20 + [2000] * 4
    batches = planner(2000).plan_batches(lengths)
    by_length = {max(lengths[i] for i in b): len(b) for b in batches}
    assert by_length[2000] == 1  # one long text fills the budget on its own
    assert max(len(b) for b in batches if max(lengths[i] for i in b) == 100) == 20


def test_one_long_text_does_not_drag_the_short_ones_up():
    # The regression this exists for: one 3836-token chunk among 561 short ones
    # tripled peak VRAM because every batch it joined was padded out to it.
    lengths = [298] * 561 + [3836]
    batches = planner(24576).plan_batches(lengths)
    worst = max(cost(b, lengths) for b in batches)
    assert worst <= 24576
    # Row batching at 32 would have cost this much for the batch it landed in.
    assert 32 * 3836 > worst * 4


def test_a_single_text_over_budget_still_gets_encoded():
    # Truncation to max_seq_length is the model's business; dropping the text
    # or looping forever is not an option.
    batches = planner(100).plan_batches([5000])
    assert batches == [[0]]


def test_an_over_budget_text_does_not_take_others_with_it():
    lengths = [5000, 10, 10]
    batches = planner(100).plan_batches(lengths)
    assert [0] in batches
    assert all(len(b) == 1 for b in batches if 0 in b)


def test_the_row_cap_still_applies():
    lengths = [1] * 100
    batches = planner(10**9, max_rows=8).plan_batches(lengths)
    assert max(len(b) for b in batches) == 8


def test_texts_are_visited_longest_first():
    # A long text must start a batch rather than join one and force the rest to
    # pad up to it.
    lengths = [10, 10, 900, 10]
    batches = planner(1000).plan_batches(lengths)
    assert batches[0] == [2]


def test_zero_length_texts_do_not_divide_by_zero():
    batches = planner(100).plan_batches([0, 0, 0])
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_an_empty_input_plans_nothing():
    assert planner(1000).plan_batches([]) == []


@pytest.mark.parametrize("budget", [1, 100, 4096, 24576, 10**6])
def test_planning_is_total_for_any_budget(budget):
    lengths = [1, 50, 345, 1313, 3836, 7000]
    batches = planner(budget).plan_batches(lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_the_plan_is_deterministic():
    lengths = [345, 3836, 298, 1313, 80]
    assert planner(4096).plan_batches(lengths) == planner(4096).plan_batches(lengths)


def test_the_measured_regression_is_bounded():
    """The 562-chunk corpus that motivated this, at the shipped default."""
    lengths = [3836] + [1313, 1060, 1014, 858] + [298] * 557
    batches = planner(24576).plan_batches(lengths)
    worst = max(cost(b, lengths) for b in batches)
    # Row batching at 32 put the 3836-token chunk in a batch costing 122752
    # padded tokens, measured at 18.55 GB of VRAM. The budget caps it.
    assert worst <= 24576
    assert 32 * 3836 / worst > 4
