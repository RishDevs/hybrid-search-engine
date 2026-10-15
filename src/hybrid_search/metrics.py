import math

import numpy as np
from scipy import stats


def ndcg_at_k(ranked, relevant, k=10):
    gains = [relevant.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g > 0)
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k=100):
    wanted = {d for d, g in relevant.items() if g > 0}
    if not wanted:
        return 0.0
    return len(wanted.intersection(ranked[:k])) / len(wanted)


def mrr_at_k(ranked, relevant, k=10):
    for i, d in enumerate(ranked[:k]):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def per_query(run, qrels, metric, k):
    fn = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k}[metric]
    return {q: fn(run.get(q, []), rel, k) for q, rel in qrels.items()}


def summarize(run, qrels):
    out = {}
    for metric, k in (("ndcg", 10), ("recall", 100), ("mrr", 10)):
        vals = per_query(run, qrels, metric, k)
        out[f"{metric}@{k}"] = float(np.mean(list(vals.values()))) if vals else 0.0
    return out


def paired_test(a, b):
    keys = sorted(a)
    x = np.array([a[q] for q in keys])
    y = np.array([b[q] for q in keys])
    if np.allclose(x, y):
        return 1.0
    return float(stats.ttest_rel(x, y).pvalue)


def to_run(results, query_ids, doc_ids, drop_self=True):
    run = {}
    for qid, (idx, _) in zip(query_ids, results):
        ranked = [doc_ids[i] for i in idx if i >= 0]
        if drop_self:
            ranked = [d for d in ranked if d != qid]
        run[qid] = ranked
    return run
