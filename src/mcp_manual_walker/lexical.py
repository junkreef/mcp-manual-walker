"""Lexical (BM25) retrieval alongside the vector index.

Dense retrieval cannot find a message identifier. Measured on this corpus with
an *exact* full scan -- no approximation involved -- the top 5 for "what does
message IEF450I mean" contained no chunk with that string in it, though 13
chunks do. The same for IEC141I (14 chunks), S0C4 (20) and S806 (3). Embeddings
put IEF450I and IEF451I in almost the same place and give a rare token no extra
weight; BM25 does the opposite, which is exactly the half that was missing.

Tokenizer choice. `unicode61` splits on non-alphanumerics and does not segment
Japanese, which sounds like a problem for a corpus queried in Japanese and is
not: the corpus is 0.00% Japanese (67 CJK characters in 3.16 million sampled),
so a Japanese analyzer would have nothing to match against. What Japanese
queries do carry is identifiers -- 12 of 20 in the evaluation set, including
every one that dense retrieval failed on -- because "メッセージ IEF450I の意味"
still spells IEF450I. Morphological analysis would add a dependency and buy
nothing measurable here. It would be the right call for a corpus with Japanese
body text; this one has none.

The index lives in the manuals database and is built after chunks are stored,
never carried in an export archive: it is derived data, and rebuilding it costs
seconds against the tens of minutes an import already takes.
"""

import logging
import re
import sqlite3
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

FTS_TABLE = "chunks_fts"

# FTS5's own vocabulary table. It reports how many documents hold a term
# straight from the index, which is the difference between a lookup and a scan:
# counting rows for a word like "system" means walking 100,338 postings, and
# doing that for every term of every query cost 350 ms a query.
VOCAB_TABLE = "chunks_vocab"

# How many chunks the index holds, recorded when it is built. Counting rows in
# an FTS5 table is a full scan -- 340 ms on this corpus -- and the rarity gate
# needs the figure on every query, so it is stored rather than recomputed.
STATS_TABLE = "chunks_fts_stats"

# Reciprocal Rank Fusion constants, one pair per retriever. The dense list gets
# the usual flat curve; the lexical list gets a steep one at a fraction of the
# weight, which is what makes this a fusion rather than a switch.
#
# Something has to give here, and it is arithmetic rather than taste. Getting
# BM25's 4th hit into an overall top 5 leaves room for at most one dense hit
# above it, so BM25's first four come too. There is no setting that surfaces a
# message definition and also keeps the dense ranking; the choice is which to
# trust on a query the gate has already judged lexical. Dense is measurably
# useless on those -- for "what does message IEF450I mean" it does not find the
# definition chunk even when the search is narrowed to the 1,437 chunks of the
# volume that contains it.
#
# What the shape does buy is a bounded tail. At LEXICAL_K=5 and weight 0.3 the
# 20th lexical hit scores 0.012, below the best dense hit at 0.016, so a long
# lexical list cannot swamp the dense one. A flat curve at higher weight
# (k=60, w=1.5) scores the same on every measure here while putting the 20th
# lexical hit above the 1st dense one, which stops being fusion at all.
RRF_K = 60
LEXICAL_K = 5
LEXICAL_WEIGHT = 0.3

# Candidates taken from each retriever before fusion.
DENSE_CANDIDATES = 20
LEXICAL_CANDIDATES = 20

# A query term is only worth asking BM25 about if it is rare. Terms appearing
# in more than this fraction of the chunks are dropped, and a query left with
# none is not sent to BM25 at all.
#
# This is not a workaround for BM25; BM25 ranks correctly. Asked for
# "what does message IEF450I mean" it puts chunks containing IEF450I at ranks
# 2 and 3, exactly as inverse document frequency should. The problem is the
# other kind of query. "how do I mount a zFS file system" contains no rare
# term at all -- its rarest is zFS, in 1,972 chunks -- and its terms together
# match 150,279 chunks, 30% of the corpus. BM25 dutifully returns a ranked 20
# of them, chosen by how densely they repeat ordinary words, and rank fusion
# has no way to tell that confident list from a grasping one: RRF sees ranks,
# never scores. Those 20 then displace dense hits that were right.
#
# Measured over 50 questions. Without this gate, identifier queries improved
# (16/50 -> 27/50 of their top 5 containing the identifier) while agreement
# with an exact vector scan collapsed from 95.6% to 59.2%. With it at 0.0005 --
# 252 chunks here -- identifier queries reach 30/50 and the 41 queries BM25
# never sees are bit-for-bit unchanged.
MAX_TERM_DOCUMENT_FREQUENCY_RATIO = 0.0005

# Floor under the ratio, because a fraction of a small collection rounds to
# nothing: at 0.0005 anything below 2,000 chunks would admit only terms
# appearing in a single chunk, which switches the lexical half off without
# saying so. Found on a 1,558-chunk single-manual database, where it made a
# dense-only run look like a regression against a hybrid one.
MIN_TERM_DOCUMENT_FREQUENCY = 25

# Query terms, split the way unicode61 splits the documents: runs of letters
# and digits, nothing else. Matching the tokenizer matters because terms are
# looked up in the FTS vocabulary -- keeping the underscore would ask it for
# "__mount", which it has never heard of, since the indexed form is "mount".
# Single characters carry no signal and match far too much.
_TERM = re.compile(r"[A-Za-z0-9]{2,}")


def sqlite_connection(session) -> sqlite3.Connection:
    """The raw sqlite3 connection under a SQLAlchemy session.

    Going through the session rather than opening the file again keeps the FTS
    writes inside the same transaction as the rows they describe, so a failed
    import cannot leave an index that disagrees with the manuals table.
    """
    return session.connection().connection.driver_connection


def create_table(conn: sqlite3.Connection) -> None:
    """Creates the FTS index if it is not there yet."""
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
        "chunk_id UNINDEXED, manual_id UNINDEXED, content, "
        "tokenize='unicode61')"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {VOCAB_TABLE} "
        f"USING fts5vocab({FTS_TABLE}, 'row')"
    )


def drop_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {STATS_TABLE}")
    conn.execute(f"DROP TABLE IF EXISTS {VOCAB_TABLE}")
    conn.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")


def add_chunks(conn: sqlite3.Connection, rows: Iterable[tuple[str, str, str]]) -> int:
    """Inserts (chunk_id, manual_id, content) triples. Returns the count."""
    written = 0
    batch: list[tuple[str, str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= 2000:
            conn.executemany(
                f"INSERT INTO {FTS_TABLE}(chunk_id, manual_id, content) "
                "VALUES (?, ?, ?)",
                batch,
            )
            written += len(batch)
            batch = []
    if batch:
        conn.executemany(
            f"INSERT INTO {FTS_TABLE}(chunk_id, manual_id, content) VALUES (?, ?, ?)",
            batch,
        )
        written += len(batch)
    return written


def optimize(conn: sqlite3.Connection) -> int:
    """Merges the FTS b-trees and records the chunk count. Call after a load."""
    conn.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES ('optimize')")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {STATS_TABLE} (total INTEGER NOT NULL)")
    conn.execute(f"DELETE FROM {STATS_TABLE}")
    total = conn.execute(f"SELECT count(*) FROM {FTS_TABLE}").fetchone()[0]
    conn.execute(f"INSERT INTO {STATS_TABLE}(total) VALUES (?)", (int(total),))
    return int(total)


def table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (FTS_TABLE,)
    ).fetchone()
    return row is not None


def build_match_query(text: str) -> Optional[str]:
    """Turns free text into an FTS5 MATCH expression, or None if nothing is left.

    A user's words cannot go to MATCH as they are: `TCP/IP` and `__mount()` are
    syntax errors, and a stray quote breaks the parse. Terms are extracted and
    each one is quoted, which both escapes them and stops FTS5 reading any of
    them as an operator.
    """
    terms = _TERM.findall(text)
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _corpus_size(conn: sqlite3.Connection) -> int:
    """The recorded chunk count, or a counted one if it was never recorded."""
    try:
        row = conn.execute(f"SELECT total FROM {STATS_TABLE}").fetchone()
        if row and row[0]:
            return int(row[0])
    except sqlite3.OperationalError:
        pass
    row = conn.execute(f"SELECT count(*) FROM {FTS_TABLE}").fetchone()
    return int(row[0]) if row else 0


def _document_frequency(conn: sqlite3.Connection, term: str) -> int:
    """How many chunks hold this term, read from the FTS vocabulary.

    fts5vocab stores the term lowercased the way the tokenizer did, so the
    lookup has to match that.
    """
    try:
        row = conn.execute(
            f"SELECT doc FROM {VOCAB_TABLE} WHERE term = ?", (term.lower(),)
        ).fetchone()
    except sqlite3.OperationalError:
        # An index built before the vocabulary table existed.
        quoted = '"' + term.replace('"', '""') + '"'
        row = conn.execute(
            f"SELECT count(*) FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ?", (quoted,)
        ).fetchone()
    return int(row[0]) if row else 0


def discriminating_terms(
    conn: sqlite3.Connection,
    text: str,
    max_df_ratio: float = MAX_TERM_DOCUMENT_FREQUENCY_RATIO,
) -> list[str]:
    """The query terms rare enough to be worth a lexical lookup.

    An empty result means the question has no lexical handle on this corpus,
    and the caller should leave the dense ranking alone rather than mix in
    twenty chunks chosen by how often they repeat ordinary words.
    """
    total = _corpus_size(conn)
    if not total:
        return []
    ceiling = max(MIN_TERM_DOCUMENT_FREQUENCY, int(total * max_df_ratio))
    kept = []
    for term in dict.fromkeys(_TERM.findall(text)):
        df = _document_frequency(conn, term)
        if 0 < df <= ceiling:
            kept.append(term)
    return kept


def search(
    conn: sqlite3.Connection,
    text: str,
    limit: int = LEXICAL_CANDIDATES,
    manual_id: Optional[str] = None,
    max_df_ratio: float = MAX_TERM_DOCUMENT_FREQUENCY_RATIO,
) -> list[str]:
    """Returns chunk ids best matching `text` by BM25, best first.

    An empty list is the honest answer for a query with no indexable term --
    a purely Japanese question against this English corpus, for instance. The
    caller then gets dense retrieval alone, which is the right outcome rather
    than a failure.
    """
    if not table_exists(conn):
        return []
    terms = discriminating_terms(conn, text, max_df_ratio)
    if not terms:
        return []
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
    sql = (
        f"SELECT chunk_id FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ?"
        + (" AND manual_id = ?" if manual_id else "")
        + f" ORDER BY bm25({FTS_TABLE}) LIMIT ?"
    )
    params: list = [match]
    if manual_id:
        params.append(manual_id)
    params.append(limit)
    try:
        return [r[0] for r in conn.execute(sql, params)]
    except sqlite3.OperationalError as exc:
        # A malformed MATCH must not take the search down; dense still answers.
        logger.warning("Lexical search failed for %r: %s", text, exc)
        return []


def rrf_fuse(
    *ranked_lists: list[str],
    k: int = RRF_K,
    weights: Optional[list[float]] = None,
    ks: Optional[list[int]] = None,
) -> list[str]:
    """Reciprocal Rank Fusion over any number of ranked id lists.

    Fusing by rank rather than by score is what makes this safe here: a cosine
    distance and a BM25 score share neither scale nor distribution, and BM25
    returns nothing at all for a query with no indexable term. Ranks are always
    comparable, and a list that is empty simply contributes nothing.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if ks is None:
        ks = [k] * len(ranked_lists)
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for weight, own_k, ranked in zip(weights, ks, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (own_k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    # Ties broken by first appearance, so the result is deterministic.
    return sorted(scores, key=lambda i: (-scores[i], first_seen[i]))


def fuse_dense_and_lexical(dense: list[str], lexical_hits: list[str]) -> list[str]:
    """The retrieval order the server uses: dense flat, lexical steep and light."""
    return rrf_fuse(
        dense,
        lexical_hits,
        weights=[1.0, LEXICAL_WEIGHT],
        ks=[RRF_K, LEXICAL_K],
    )
