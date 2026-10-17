import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from .fusion import top_margin

FEATURES = (
    "fused_margin",
    "dense_top1",
    "dense_gap_1_2",
    "dense_gap_1_10",
    "overlap_at_10",
    "bm25_gap_1_2",
    "log_bm25_hits",
    "query_terms",
)


def _at(arr, i):
    return float(arr[i]) if len(arr) > i else 0.0


def query_features(n_terms, bm_ids, bm_scores, dense_ids, dense_scores, fused_scores):
    b1, b2 = _at(bm_scores, 0), _at(bm_scores, 1)
    d1 = _at(dense_scores, 0)
    shared = len(set(np.asarray(bm_ids[:10]).tolist()) & set(np.asarray(dense_ids[:10]).tolist()))
    return np.array(
        [
            top_margin(fused_scores),
            d1,
            d1 - _at(dense_scores, 1),
            d1 - _at(dense_scores, 9),
            shared / 10,
            (b1 - b2) / b1 if b1 > 0 else 0.0,
            math.log1p(len(bm_ids)),
            float(n_terms),
        ]
    )


def _ridge(X, y, lam):
    mean, scale = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mean) / scale, np.ones((len(X), 1))])
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return mean, scale, w


def pick_threshold(pred, base, full, recover):
    gain = full.mean() - base.mean()
    if gain <= 0:
        return math.inf, 0.0
    target = base.sum() + recover * gain * len(base)
    total = base.sum()
    for k, i in enumerate(np.argsort(-pred, kind="stable"), start=1):
        total += full[i] - base[i]
        if total >= target:
            return float(pred[i]), k / len(base)
    return -math.inf, 1.0


@dataclass
class Router:
    mean: list
    scale: list
    weights: list
    threshold: float
    recover: float = 0.8
    calib_share: float = 1.0

    def predict(self, X):
        X = np.atleast_2d(X)
        w = np.asarray(self.weights)
        return (X - np.asarray(self.mean)) / np.asarray(self.scale) @ w[:-1] + w[-1]

    def should_rerank(self, features):
        return bool(self.predict(features)[0] >= self.threshold)

    @classmethod
    def always(cls):
        n = len(FEATURES)
        return cls([0.0] * n, [1.0] * n, [0.0] * (n + 1), -math.inf)

    @classmethod
    def train(cls, X, base, full, recover=0.8, lam=10.0, folds=5, seed=0):
        X, base, full = np.asarray(X, float), np.asarray(base, float), np.asarray(full, float)
        gain = full - base
        which = np.random.default_rng(seed).integers(0, folds, len(X))
        held_pred = np.empty(len(X))
        for k in range(folds):
            fit = which != k
            mean, scale, w = _ridge(X[fit], gain[fit], lam)
            held_pred[~fit] = (X[~fit] - mean) / scale @ w[:-1] + w[-1]
        threshold, share = pick_threshold(held_pred, base, full, recover)
        mean, scale, w = _ridge(X, gain, lam)
        return cls(mean.tolist(), scale.tolist(), w.tolist(), threshold, recover, share)

    def save(self, path):
        path.write_text(json.dumps({**asdict(self), "features": FEATURES}, indent=2))

    @classmethod
    def load(cls, path):
        raw = json.loads(path.read_text())
        raw.pop("features", None)
        return cls(**raw)
