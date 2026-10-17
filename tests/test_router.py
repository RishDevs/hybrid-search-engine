import math

import numpy as np
import pytest

from hybrid_search.router import FEATURES, Router, pick_threshold, query_features


def test_features_shape_and_values():
    x = query_features(
        n_terms=3,
        bm_ids=np.array([1, 2, 3]),
        bm_scores=np.array([10.0, 5.0, 1.0]),
        dense_ids=np.array([2, 9]),
        dense_scores=np.array([0.8, 0.6]),
        fused_scores=np.array([0.04, 0.03]),
    )
    assert x.shape == (len(FEATURES),)
    margin, d1, gap12, gap110, overlap, bm_gap, log_hits, terms = x
    assert margin == pytest.approx(0.25)
    assert d1 == pytest.approx(0.8) and gap12 == pytest.approx(0.2) and gap110 == pytest.approx(0.8)
    assert overlap == pytest.approx(0.1)
    assert bm_gap == pytest.approx(0.5)
    assert log_hits == pytest.approx(math.log1p(3)) and terms == 3


def test_features_with_no_bm25_hits():
    x = query_features(0, np.array([], dtype=int), np.array([]), np.array([4]), np.array([0.5]), np.array([0.01]))
    assert np.isfinite(x).all()


def test_pick_threshold_takes_highest_predicted_gain_first():
    pred = np.array([0.9, 0.1, 0.5, -0.2])
    base = np.array([0.0, 0.5, 0.2, 0.5])
    full = np.array([1.0, 0.5, 0.4, 0.4])
    threshold, share = pick_threshold(pred, base, full, recover=0.8)
    assert threshold == 0.9 and share == 0.25


def test_pick_threshold_when_reranker_hurts():
    assert pick_threshold(np.array([1.0, 2.0]), np.array([0.5, 0.5]), np.array([0.4, 0.4]), 0.8) == (math.inf, 0.0)


def test_router_learns_which_queries_gain():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, len(FEATURES)))
    gain = np.where(X[:, 4] < -0.5, 0.3, 0.0)
    base = rng.uniform(0.2, 0.6, 400)
    r = Router.train(X, base, base + gain, recover=0.9)
    sent = r.predict(X) >= r.threshold
    assert sent[X[:, 4] < -0.5].mean() > 0.9
    assert sent.mean() < 0.6


def test_always_reranks_and_round_trip(tmp_path):
    r = Router.always()
    assert r.should_rerank(np.zeros(len(FEATURES)))
    path = tmp_path / "router.json"
    r.save(path)
    again = Router.load(path)
    assert again.threshold == -math.inf and again.should_rerank(np.ones(len(FEATURES)))
