"""The lexical half of retrieval, and the fusion that keeps it from doing harm."""

import sqlite3

import pytest

from mcp_manual_walker.lexical import (
    FTS_TABLE,
    add_chunks,
    build_match_query,
    create_table,
    discriminating_terms,
    fuse_dense_and_lexical,
    optimize,
    rrf_fuse,
    search,
    table_exists,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    create_table(c)
    add_chunks(
        c,
        [
            ("c1", "m1", "IEF450I JOB FAILED. The job step was cancelled by the operator."),
            ("c2", "m1", "IEF451I is a different message about job restart."),
            ("c3", "m1", "Steps for mounting file systems. The mount point must be empty."),
            ("c4", "m2", "The __mount() function makes a file system available."),
            ("c5", "m2", "Configure TCP/IP with the PROFILE data set."),
            ("c6", "m2", "カップリング・ファシリティの構成について"),
        ],
    )
    # Filler carrying the ordinary words, so their inverse document frequency
    # is low and a rare identifier can out-rank them. Without it every word in
    # a six-document fixture is equally rare and BM25 has nothing to weigh.
    add_chunks(
        c,
        [
            (f"f{i}", "m3",
             f"This message describes a job step and the data set it uses ({i}).")
            for i in range(60)
        ],
    )
    optimize(c)
    return c


def test_a_rare_identifier_is_found_where_dense_retrieval_missed_it(conn):
    """The case the whole module exists for: an exact scan of the vectors
    returned nothing containing IEF450I, though the corpus holds 13 such chunks."""
    assert search(conn, "what does message IEF450I mean")[0] == "c1"


def test_a_near_miss_identifier_is_not_confused_with_its_neighbour(conn):
    """IEF450I and IEF451I sit almost on top of each other in embedding space."""
    hits = search(conn, "IEF450I")
    assert hits == ["c1"]


def test_a_query_is_scoped_to_one_manual(conn):
    assert search(conn, "mount point", manual_id="m1", max_df_ratio=1.0) == ["c3"]
    assert search(conn, "mount point", manual_id="m2", max_df_ratio=1.0) == ["c4"]


def test_slashes_and_punctuation_do_not_break_the_match_syntax(conn):
    # Passed to MATCH unquoted, "TCP/IP" and "__mount()" are syntax errors.
    assert search(conn, "TCP/IP profile", max_df_ratio=1.0)[0] == "c5"
    # unicode61 indexes "__mount()" as "mount", so this also reaches the
    # chunk about mounting -- correct behaviour, just not an exact-match tool.
    assert search(conn, "__mount() return codes", max_df_ratio=1.0)[0] == "c4"


def test_a_quote_in_the_query_cannot_break_the_parse(conn):
    # Neither of these may raise, and neither may be read as FTS5 syntax.
    assert "c3" in search(conn, 'mount " OR content MATCH "x', max_df_ratio=1.0)
    assert search(conn, '"""', max_df_ratio=1.0) == []


def test_a_purely_japanese_question_matches_nothing(conn):
    """Not a failure. The corpus is 0.00% Japanese, so there is nothing to
    match; the caller falls back to dense retrieval, which handles these."""
    assert build_match_query("カップリング・ファシリティとは何か") is None
    assert search(conn, "カップリング・ファシリティとは何か") == []


def test_a_japanese_question_carrying_an_identifier_still_matches(conn):
    """12 of the 20 Japanese evaluation queries carry one; identifiers are
    never translated."""
    assert search(conn, "メッセージ IEF450I の意味") == ["c1"]


def test_single_characters_are_not_terms():
    assert build_match_query("a b c") is None
    assert build_match_query("a bc") == '"bc"'


def test_search_without_the_table_returns_nothing():
    c = sqlite3.connect(":memory:")
    assert not table_exists(c)
    assert search(c, "IEF450I") == []


def test_add_chunks_reports_what_it_wrote(conn):
    before = conn.execute(f"SELECT count(*) FROM {FTS_TABLE}").fetchone()[0]
    n = add_chunks(conn, [(f"x{i}", "m4", f"body {i}") for i in range(2500)])
    assert n == 2500
    assert (
        conn.execute(f"SELECT count(*) FROM {FTS_TABLE}").fetchone()[0] == before + 2500
    )


def test_rrf_prefers_what_both_retrievers_agree_on():
    dense = ["a", "b", "c"]
    lexical = ["c", "d", "a"]
    assert rrf_fuse(dense, lexical)[0] == "a"


def test_rrf_with_an_empty_list_is_the_other_list():
    dense = ["a", "b", "c"]
    assert rrf_fuse(dense, []) == dense


def test_rrf_keeps_a_hit_only_one_retriever_found():
    """A message id that only BM25 can reach must survive fusion."""
    assert "z" in rrf_fuse(["a", "b"], ["z"])


def test_rrf_is_deterministic_on_ties():
    # Same score for both; first appearance wins, every time.
    assert rrf_fuse(["a"], ["b"]) == ["a", "b"]
    assert rrf_fuse(["b"], ["a"]) == ["b", "a"]


def test_only_rare_terms_are_asked_of_bm25(conn):
    """The gate that keeps fusion honest.

    BM25 ranks correctly -- given "what does message IEF450I mean" over the
    real corpus it puts IEF450I chunks at ranks 2 and 3, which is inverse
    document frequency working as designed. The damage comes from questions
    with no rare term at all: their terms match a third of the corpus, BM25
    returns a ranked 20 chosen by how often ordinary words repeat, and RRF --
    which sees ranks and never scores -- cannot tell that list from a
    confident one.
    """
    # "message" and "job" are in the filler; the identifier is in one chunk.
    assert discriminating_terms(conn, "IEF450I", max_df_ratio=0.05) == ["IEF450I"]
    assert discriminating_terms(conn, "message job step", max_df_ratio=0.05) == []


def test_a_question_with_no_rare_term_is_not_sent_to_bm25(conn):
    """An empty result leaves the dense ranking untouched, which is correct:
    fusing noise at full rank authority is what cost 43 points of agreement
    with an exact vector scan."""
    assert search(conn, "this message describes a job step", max_df_ratio=0.05) == []


def test_the_rare_term_survives_among_common_ones(conn):
    """A real question mixes both; only the rare half should reach BM25."""
    assert discriminating_terms(
        conn, "what does message IEF450I mean", max_df_ratio=0.05
    ) == ["IEF450I"]
    assert search(conn, "what does message IEF450I mean", max_df_ratio=0.05) == ["c1"]


def test_fusion_lets_a_good_lexical_hit_outrank_the_dense_head():
    """The point of the weighted, steep lexical curve.

    An identifier's definition sits at BM25 rank 4 once the search is narrowed
    to the right manual, and has to clear the dense top few to be seen at all.
    """
    dense = [f"d{i}" for i in range(1, 21)]
    lex = ["x1", "x2", "x3", "defn"] + [f"y{i}" for i in range(5, 21)]
    top5 = fuse_dense_and_lexical(dense, lex)[:5]
    assert "defn" in top5


def test_fusion_does_not_let_the_lexical_tail_swamp_the_dense_head():
    """A flat curve at higher weight scores the same on the corpus and puts the
    20th lexical hit above the 1st dense one, which is no longer fusion."""
    dense = [f"d{i}" for i in range(1, 21)]
    lex = [f"y{i}" for i in range(1, 21)]
    order = fuse_dense_and_lexical(dense, lex)
    assert order.index("d1") < order.index("y20")


def test_fusion_of_an_empty_lexical_list_is_the_dense_order():
    dense = [f"d{i}" for i in range(1, 21)]
    assert fuse_dense_and_lexical(dense, []) == dense
