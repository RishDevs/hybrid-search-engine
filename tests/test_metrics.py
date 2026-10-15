import math

import pytest

from hybrid_search import metrics


def test_ndcg_perfect_and_partial():
    rel = {"a": 2, "b": 1}
    assert metrics.ndcg_at_k(["a", "b", "c"], rel) == pytest.approx(1.0)
    got = metrics.ndcg_at_k(["c", "a"], rel)
    ideal = 2 + 1 / math.log2(3)
    assert got == pytest.approx((2 / math.log2(3)) / ideal)


def test_ndcg_ignores_zero_graded_judgements():
    assert metrics.ndcg_at_k(["x"], {"x": 0}) == 0.0


def test_recall_and_mrr():
    rel = {"a": 1, "b": 1, "z": 0}
    assert metrics.recall_at_k(["a", "c"], rel, 100) == 0.5
    assert metrics.mrr_at_k(["c", "b"], rel) == 0.5


def test_to_run_drops_query_matching_its_own_id():
    import numpy as np

    run = metrics.to_run([(np.array([0, 1]), None)], ["q1"], ["q1", "d2"])
    assert run == {"q1": ["d2"]}


def test_paired_test_identical_runs():
    assert metrics.paired_test({"q": 0.5}, {"q": 0.5}) == 1.0
