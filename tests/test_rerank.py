import numpy as np

from hybrid_search.rerank import Reranker, ScoreCache


class CountingModel:
    def __init__(self):
        self.seen = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        pairs = list(pairs)
        self.seen.extend(pairs)
        return [float(len(q) * 10 + len(p)) for q, p in pairs]


def fake_reranker(tmp_path):
    r = Reranker.__new__(Reranker)
    r.name, r.batch_size, r.device = "fake", 8, "cpu"
    r.model = CountingModel()
    r.cache = ScoreCache("fake", tmp_path)
    return r


def test_cache_scores_each_pair_once(tmp_path):
    r = fake_reranker(tmp_path)
    first = r.score("q", ["aa", "b", "aa"])
    assert first.tolist() == [12.0, 11.0, 12.0]
    assert r.model.seen == [("q", "aa"), ("q", "b")]
    again = r.score("q", ["b", "aa", "ccc"])
    assert again.tolist() == [11.0, 12.0, 13.0]
    assert r.model.seen[2:] == [("q", "ccc")]


def test_cache_survives_reopening(tmp_path):
    fake_reranker(tmp_path).score("query", ["doc"])
    r = fake_reranker(tmp_path)
    assert r.score("query", ["doc"]).tolist() == [53.0]
    assert r.model.seen == []


def test_rerank_many_splits_and_sorts_per_query(tmp_path):
    r = fake_reranker(tmp_path)
    texts = ["x", "xxx", "xx"]
    out = r.rerank_many(["a", "bb"], [[0, 1, 2], [2, 0]], texts)
    np.testing.assert_array_equal(out[0][0], [1, 2, 0])
    np.testing.assert_array_equal(out[1][0], [2, 0])
    assert out[1][1].tolist() == [22.0, 21.0]


def test_key_separates_query_and_passage():
    assert ScoreCache.key("ab", "c") != ScoreCache.key("a", "bc")
