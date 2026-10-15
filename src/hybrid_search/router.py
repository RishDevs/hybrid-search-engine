import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MarginRouter:
    threshold: float
    recover: float = 0.9
    calib_fraction: float = 1.0

    def should_rerank(self, margin):
        return margin < self.threshold

    @classmethod
    def always(cls):
        return cls(threshold=math.inf)

    @classmethod
    def calibrate(cls, margins, fused_ndcg, reranked_ndcg, recover=0.9):
        m = np.asarray(margins, dtype=np.float64)
        base = np.asarray(fused_ndcg, dtype=np.float64)
        full = np.asarray(reranked_ndcg, dtype=np.float64)
        gain = full.mean() - base.mean()
        if gain <= 0:
            return cls(0.0, recover, 0.0)
        target = base.mean() + recover * gain

        for t in np.unique(np.append(m, math.inf)):
            if np.where(m < t, full, base).mean() >= target:
                return cls(float(t), recover, float((m < t).mean()))
        return cls(math.inf, recover, 1.0)

    def save(self, path):
        raw = asdict(self)
        raw["threshold"] = None if math.isinf(self.threshold) else self.threshold
        path.write_text(json.dumps(raw, indent=2))

    @classmethod
    def load(cls, path):
        raw = json.loads(path.read_text())
        if raw["threshold"] is None:
            raw["threshold"] = math.inf
        return cls(**raw)
