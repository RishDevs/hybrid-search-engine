import numpy as np


def rrf(rankings, k=60, weights=None, limit=None):
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for ranking, w in zip(rankings, weights):
        if not w:
            continue
        for rank, doc in enumerate(ranking, start=1):
            doc = int(doc)
            if doc < 0:
                continue
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit:
        ordered = ordered[:limit]
    ids = np.fromiter((d for d, _ in ordered), dtype=np.int64, count=len(ordered))
    scores = np.fromiter((s for _, s in ordered), dtype=np.float32, count=len(ordered))
    return ids, scores


def top_margin(scores):
    if len(scores) < 2 or scores[0] <= 0:
        return 1.0 if len(scores) == 1 else 0.0
    return float((scores[0] - scores[1]) / scores[0])
