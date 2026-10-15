import json
import math

import numpy as np

from hybrid_search.router import MarginRouter


def test_calibrate_reranks_only_the_low_margin_queries_it_needs():
    margins = np.array([0.0, 0.1, 0.5, 0.9])
    fused = np.array([0.2, 0.4, 0.8, 1.0])
    full = np.array([0.8, 0.6, 0.8, 1.0])
    r = MarginRouter.calibrate(margins, fused, full, recover=0.7)
    assert r.threshold == 0.1
    assert r.should_rerank(0.05) and not r.should_rerank(0.1)
    assert r.calib_fraction == 0.25


def test_full_recovery_needs_every_helpful_query():
    margins = np.array([0.0, 0.1, 0.5, 0.9])
    fused = np.array([0.2, 0.4, 0.8, 1.0])
    full = np.array([0.8, 0.6, 0.8, 1.0])
    assert MarginRouter.calibrate(margins, fused, full, recover=0.95).threshold == 0.5


def test_never_reranks_when_reranker_does_not_help():
    r = MarginRouter.calibrate([0.1, 0.2], [0.5, 0.5], [0.4, 0.5])
    assert r.threshold == 0.0 and r.calib_fraction == 0.0
    assert not r.should_rerank(0.0)


def test_round_trip(tmp_path):
    path = tmp_path / "router.json"
    MarginRouter.always().save(path)
    assert json.loads(path.read_text())["threshold"] is None
    loaded = MarginRouter.load(path)
    assert math.isinf(loaded.threshold) and loaded.should_rerank(123.0)
