import math

import numpy as np
import pytest

from hybrid_search.bm25 import BM25, Analyzer


def test_analyzer_drops_stopwords_and_stems():
    assert Analyzer()("The runners are running to the store") == ["runner", "run", "store"]


def test_scores_match_hand_computed_formula():
    docs = ["apple banana", "apple apple cherry", "cherry date elder fig"]
    bm = BM25(k1=1.2, b=0.75).fit(docs)

    n, avg = 3, (2 + 3 + 4) / 3
    idf = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    tf, dl = 2, 3
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avg))

    scores = bm.score("apple")
    assert scores[1] == pytest.approx(expected, rel=1e-5)
    assert scores[2] == 0
    assert scores[1] > scores[0]


def test_rare_terms_outweigh_common_ones():
    bm = BM25().fit(["common rare", "common", "common", "common"])
    s = bm.score("common rare")
    assert np.argmax(s) == 0


def test_search_skips_zero_scores_and_unknown_terms():
    bm = BM25().fit(["alpha beta", "gamma delta"])
    ids, scores = bm.search("alpha zzzz", k=10)
    assert ids.tolist() == [0]
    empty_ids, _ = bm.search("zzzz", k=10)
    assert len(empty_ids) == 0


def test_save_and_load_round_trip(tmp_path):
    docs = ["one two three", "two three four", "four five six"]
    bm = BM25().fit(docs)
    path = tmp_path / "bm25.npz"
    bm.save(path)
    again = BM25.load(path)
    assert again.k1 == bm.k1 and again.b == bm.b
    np.testing.assert_allclose(again.score("two four"), bm.score("two four"))
