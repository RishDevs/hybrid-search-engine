import json
import platform
import random
import subprocess
import time

import numpy as np

from . import data, metrics
from .config import RESULTS_DIR, Config, artifact_dir
from .embed import Encoder, cached_doc_vectors
from .pipeline import HybridSearcher

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid+rerank")


def environment():
    import faiss
    import sentence_transformers
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip() or cpu
        except OSError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "faiss": faiss.__version__,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "commit": commit,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def evaluate(dataset, cfg=None, out_dir=None):
    cfg = cfg or Config()
    out_dir = out_dir or RESULTS_DIR / "eval"
    searcher = HybridSearcher.open(dataset, cfg, cache_scores=True)
    ds = searcher.ds
    qids = list(ds.queries)
    texts = [ds.queries[q] for q in qids]

    started = time.perf_counter()
    runs = searcher.run_batch(texts, progress=True)
    elapsed = time.perf_counter() - started

    systems, ndcg = {}, {}
    for name, results in runs.items():
        if name == "features":
            continue
        run = metrics.to_run(results, qids, ds.doc_ids)
        systems[name] = metrics.summarize(run, ds.qrels)
        ndcg[name] = metrics.per_query(run, ds.qrels, "ndcg", 10)

    final = "hybrid+rerank"
    significance = {
        f"{final} vs {other}": metrics.paired_test(ndcg[final], ndcg[other])
        for other in ndcg
        if other != final
    }
    significance["hybrid vs dense"] = metrics.paired_test(ndcg["hybrid"], ndcg["dense"])

    report = {
        "dataset": dataset,
        "docs": len(ds),
        "queries": len(qids),
        "config": cfg.to_dict(),
        "systems": systems,
        "p_values": significance,
        "seconds": round(elapsed, 1),
        "env": environment(),
    }
    write_json(out_dir / f"{dataset}.json", report)
    write_json(
        out_dir / f"{dataset}.per_query.json",
        {
            "query_ids": qids,
            "features": runs["features"].tolist(),
            "ndcg": {name: [vals[q] for q in qids] for name, vals in ndcg.items()},
        },
    )
    return report


TUNE_SPLITS = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}
RERANKERS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "Alibaba-NLP/gte-reranker-modernbert-base",
)


def _held_out_runs(cfg, splits, rerank=False):
    prepared = []
    for name, split in splits.items():
        searcher = HybridSearcher.open(name, cfg, with_reranker=rerank, cache_scores=True)
        held = data.load(name, split=split)
        qids = list(held.queries)
        texts = [held.queries[q] for q in qids]
        runs = searcher.run_batch(texts, rerank=rerank, progress=rerank)
        prepared.append((name, searcher, held.qrels, qids, texts, runs))
    return prepared


def _ndcg_per_query(results, doc_ids, qrels, qids):
    run = metrics.to_run(results, qids, doc_ids)
    vals = metrics.per_query(run, qrels, "ndcg", 10)
    return np.array([vals[q] for q in qids])


def tune(cfg=None, splits=None, rerankers=RERANKERS, depths=(10, 20, 30),
         bm25_grid=(0.1, 0.2, 0.3, 0.5), rerank_grid=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0)):
    from .fusion import rrf
    from .rerank import Reranker

    cfg = cfg or Config()
    splits = splits or TUNE_SPLITS
    prepared = _held_out_runs(cfg, splits)

    pools = []
    for _, _, _, _, _, runs in prepared:
        per_query = []
        for b, d in zip(runs["bm25"], runs["dense"]):
            ids = set()
            for wb in bm25_grid:
                ids.update(rrf([b[0], d[0]], cfg.rrf_k, [wb, cfg.w_dense], max(depths))[0].tolist())
            per_query.append(sorted(ids))
        pools.append(per_query)

    dense_only = {
        name: float(_ndcg_per_query(runs["dense"], s.ds.doc_ids, qrels, qids).mean())
        for name, s, qrels, qids, _, runs in prepared
    }

    grid = []
    for model in rerankers:
        reranker = Reranker(model, cfg.device, cache=True)
        scored = []
        for (_, s, _, _, texts, _), pool in zip(prepared, pools):
            ranked = reranker.rerank_many(texts, pool, s.ds.texts, progress=True)
            scored.append([dict(zip(ids.tolist(), sc.tolist())) for ids, sc in ranked])
        for depth in depths:
            for wb in bm25_grid:
                for wr in rerank_grid:
                    scores = {}
                    for (name, s, qrels, qids, _, runs), ce in zip(prepared, scored):
                        final = []
                        for b, d, sc in zip(runs["bm25"], runs["dense"], ce):
                            cand = rrf([b[0], d[0]], cfg.rrf_k, [wb, cfg.w_dense], depth)[0].tolist()
                            ce_ids = sorted(cand, key=lambda i: -sc[i])
                            final.append(rrf([ce_ids, b[0], d[0]], cfg.rrf_k, [wr, wb, cfg.w_dense]))
                        scores[name] = float(_ndcg_per_query(final, s.ds.doc_ids, qrels, qids).mean())
                    grid.append({"reranker": model, "rerank_depth": depth, "w_bm25": wb, "w_rerank": wr,
                                 "ndcg@10": scores, "mean": float(np.mean(list(scores.values())))})
        top = max((g for g in grid if g["reranker"] == model), key=lambda g: g["mean"])
        print(f"{model:44s} best depth={top['rerank_depth']} w_bm25={top['w_bm25']} w_rerank={top['w_rerank']} "
              f"mean dev nDCG@10 {top['mean'] * 100:.2f}", flush=True)

    best = max(grid, key=lambda g: g["mean"])
    report = {
        "splits": splits,
        "dense_only": dense_only,
        "dense_only_mean": float(np.mean(list(dense_only.values()))),
        "best": best,
        "best_per_reranker": {m: max((g for g in grid if g["reranker"] == m), key=lambda g: g["mean"]) for m in rerankers},
        "grid": grid,
        "config": cfg.to_dict(),
    }
    write_json(RESULTS_DIR / "tune.json", report)
    return report


def _latency(searcher, queries, mode, warmup=3):
    for q in queries[:warmup]:
        searcher.search(q, rerank=mode)
    samples, reranked = [], 0
    for q in queries:
        res = searcher.search(q, rerank=mode)
        samples.append(res.timings_ms["total"])
        reranked += res.reranked
    arr = np.array(samples)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "reranked_share": round(reranked / len(queries), 4),
    }


def _routing_quality(base, full, sent):
    routed = np.where(sent, full, base).mean()
    share = float(sent.mean())
    gain = full.mean() - base.mean()
    return {
        "queries": int(len(base)),
        "reranked_share": share,
        "ndcg_hybrid": float(base.mean()),
        "ndcg_full_rerank": float(full.mean()),
        "ndcg_routed": float(routed),
        "ndcg_random_same_share": float(base.mean() + share * gain),
        "gain_recovered": float((routed - base.mean()) / gain) if gain > 0 else None,
    }


def routing(cfg=None, recover=0.8, latency_queries=100, seed=7, out_dir=None):
    from .config import ROUTER_FILE
    from .router import FEATURES, Router

    cfg = cfg or Config()
    out_dir = out_dir or RESULTS_DIR / "routing"

    X, base, full = [], [], []
    for _, s, qrels, qids, _, runs in _held_out_runs(cfg, TUNE_SPLITS, rerank=True):
        X.append(runs["features"])
        base.append(_ndcg_per_query(runs["hybrid"], s.ds.doc_ids, qrels, qids))
        full.append(_ndcg_per_query(runs["hybrid+rerank"], s.ds.doc_ids, qrels, qids))
    router = Router.train(np.vstack(X), np.concatenate(base), np.concatenate(full), recover)
    router.save(ROUTER_FILE)

    test = {}
    pooled_base, pooled_full, pooled_sent = [], [], []
    for name in TUNE_SPLITS:
        pq_file = RESULTS_DIR / "eval" / f"{name}.per_query.json"
        if not pq_file.exists():
            evaluate(name, cfg)
        pq = json.loads(pq_file.read_text())
        b = np.array(pq["ndcg"]["hybrid"])
        f = np.array(pq["ndcg"]["hybrid+rerank"])
        sent = router.predict(np.array(pq["features"])) >= router.threshold
        test[name] = _routing_quality(b, f, sent)
        pooled_base.append(b)
        pooled_full.append(f)
        pooled_sent.append(sent)

    rng = random.Random(seed)
    latency = {}
    for name in TUNE_SPLITS:
        searcher = HybridSearcher.open(name, cfg)
        searcher.router = router
        qids = rng.sample(sorted(searcher.ds.queries), min(latency_queries, len(searcher.ds.queries)))
        texts = [searcher.ds.queries[q] for q in qids]
        lat = {mode: _latency(searcher, texts, mode) for mode in ("never", "always", "auto")}
        lat["mean_cut"] = 1 - lat["auto"]["mean_ms"] / lat["always"]["mean_ms"]
        lat["device"] = searcher.reranker.device
        latency[name] = lat
        write_json(out_dir / f"{name}.json", {"dataset": name, "test": test[name], "latency": lat})

    pooled = _routing_quality(np.concatenate(pooled_base), np.concatenate(pooled_full), np.concatenate(pooled_sent))
    always = np.mean([latency[n]["always"]["mean_ms"] for n in latency])
    auto = np.mean([latency[n]["auto"]["mean_ms"] for n in latency])
    summary = {
        "recover_target": recover,
        "threshold": router.threshold,
        "dev_share": router.calib_share,
        "weights": {"intercept": router.weights[-1], **dict(zip(FEATURES, router.weights[:-1]))},
        "test_pooled": pooled,
        "test": test,
        "latency_mean_ms": {"always": float(always), "auto": float(auto), "cut": float(1 - auto / always)},
        "env": environment(),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


BENCH_CONFIGS = [
    ("Flat", None, [None]),
    ("SQfp16", None, [None]),
    ("SQ8", None, [None]),
    ("PQ48", None, [None]),
    ("OPQ48,PQ48", None, [None]),
    ("HNSW32,Flat", "efSearch", [16, 24, 32, 48, 64, 96, 128, 256, 512]),
    ("HNSW32,SQ8", "efSearch", [16, 24, 32, 48, 64, 96, 128, 256, 512]),
    ("IVF1024,Flat", "nprobe", [1, 2, 4, 8, 12, 16, 24, 32, 64, 128]),
    ("IVF4096,Flat", "nprobe", [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256]),
    ("IVF1024,SQ8", "nprobe", [1, 2, 4, 8, 12, 16, 24, 32, 64, 128]),
    ("IVF1024,PQ48", "nprobe", [1, 2, 4, 8, 16, 32, 64, 128]),
    ("BFlat", None, [None]),
]

RESCORED = {"PQ48", "OPQ48,PQ48", "BFlat"}


def _recall(found, truth):
    k = truth.shape[1]
    hits = sum(len(set(f[:k]) & set(t)) for f, t in zip(found, truth))
    return hits / truth.size


def _single_query_ms(index, queries, k):
    for q in queries[:20]:
        index.search(q, k)
    start = time.perf_counter()
    for q in queries:
        index.search(q, k)
    return (time.perf_counter() - start) * 1000 / len(queries)


def _ndcg(ds, qids, ids):
    run = metrics.to_run([(row, None) for row in ids], qids, ds.doc_ids)
    return float(np.mean(list(metrics.per_query(run, ds.qrels, "ndcg", 10).values())))


def bench(dataset="quora-200k", cfg=None, target=0.95, max_loss=0.5, latency_queries=1000, configs=None, out_dir=None):
    from .dense import DenseIndex, rescore, single_threaded

    cfg = cfg or Config()
    out_dir = out_dir or RESULTS_DIR / "bench"
    ds = data.load(dataset)
    encoder = Encoder(cfg.dense_model, cfg.query_prefix, cfg.device)
    vectors = cached_doc_vectors(encoder, ds, artifact_dir(dataset, cfg))
    vectors = np.ascontiguousarray(vectors)
    qids = list(ds.queries)
    qvecs = encoder.encode_queries([ds.queries[q] for q in qids])
    timing_q = qvecs[np.random.default_rng(0).choice(len(qvecs), min(latency_queries, len(qvecs)), replace=False)]
    timing_q = [timing_q[i : i + 1] for i in range(len(timing_q))]

    exact = DenseIndex.build(vectors, "Flat")
    _, exact_ids = exact.search(qvecs, 100)
    truth = exact_ids[:, :10]
    exact_ndcg = _ndcg(ds, qids, exact_ids)

    rows = []
    for factory, knob, values in configs or BENCH_CONFIGS:
        t0 = time.perf_counter()
        index = DenseIndex.build(vectors, factory)
        build_s = time.perf_counter() - t0
        size = index.nbytes()

        sweep = []
        for v in values:
            if knob:
                index.tune(**{knob: v})
            _, found = index.search(qvecs, 100)
            with single_threaded():
                ms = _single_query_ms(index, timing_q, 10)
            sweep.append({
                "param": v,
                "recall@10": _recall(found[:, :10], truth),
                "latency_ms": ms,
                "ndcg@10": _ndcg(ds, qids, found),
            })

        reached = [s for s in sweep if s["recall@10"] >= target]
        best = min(reached, key=lambda s: s["latency_ms"]) if reached else max(sweep, key=lambda s: s["recall@10"])
        close = [s for s in sweep if s["ndcg@10"] >= exact_ndcg - max_loss / 100]
        quality = min(close, key=lambda s: s["latency_ms"]) if close else None
        if knob:
            index.tune(**{knob: best["param"]})

        row = {
            "index": factory,
            "knob": knob,
            "build_s": round(build_s, 2),
            "bytes": size,
            "bytes_per_vector": round(size / len(ds), 1),
            "compression": round(vectors.nbytes / size, 1),
            "operating_point": best,
            "reaches_target": bool(reached),
            "ndcg@10": best["ndcg@10"],
            "quality_point": quality,
            "sweep": sweep,
        }
        if factory in RESCORED:
            _, cand = index.search(qvecs, 100)
            _, reranked = rescore(qvecs, cand, vectors)
            row["rescored"] = {
                "recall@10": _recall(reranked[:, :10], truth),
                "ndcg@10": _ndcg(ds, qids, reranked),
                "latency_ms": _rescore_ms(index, timing_q, vectors),
            }
        rows.append(row)
        print(f"{factory:14s} {row['bytes_per_vector']:8.1f} B/vec  recall {best['recall@10']:.3f}  "
              f"{best['latency_ms']:.3f} ms  ndcg {row['ndcg@10']*100:.2f}", flush=True)

    flat = rows[0]
    flat_ms = flat["operating_point"]["latency_ms"]
    for row in rows:
        row["speedup_vs_flat"] = flat_ms / row["operating_point"]["latency_ms"]
        if row["quality_point"]:
            row["quality_point"]["speedup_vs_flat"] = flat_ms / row["quality_point"]["latency_ms"]
        row["ndcg_delta_points"] = (row["ndcg@10"] - flat["ndcg@10"]) * 100
        if "rescored" in row:
            row["rescored"]["ndcg_delta_points"] = (row["rescored"]["ndcg@10"] - flat["ndcg@10"]) * 100

    report = {
        "dataset": dataset,
        "docs": len(ds),
        "queries": len(qids),
        "dim": int(vectors.shape[1]),
        "recall_target": target,
        "max_ndcg_loss_points": max_loss,
        "latency_queries": len(timing_q),
        "threads_for_latency": 1,
        "model": cfg.dense_model,
        "indexes": rows,
        "env": environment(),
    }
    write_json(out_dir / f"{dataset}.json", report)
    plot_bench(report, out_dir / f"{dataset}.png")
    return report


def _rescore_ms(index, queries, vectors):
    from .dense import rescore, single_threaded

    with single_threaded():
        start = time.perf_counter()
        for q in queries:
            _, cand = index.search(q, 100)
            rescore(q, cand, vectors)
        return (time.perf_counter() - start) * 1000 / len(queries)


def plot_bench(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.get_cmap("tab20").colors
    for i, row in enumerate(report["indexes"]):
        xs = [s["latency_ms"] for s in row["sweep"]]
        ys = [s["recall@10"] for s in row["sweep"]]
        c = colors[i % len(colors)]
        left.plot(xs, ys, marker="o", ms=3, label=row["index"], color=c)
        right.scatter(row["bytes_per_vector"], row["ndcg@10"] * 100, color=c)
        right.annotate(row["index"], (row["bytes_per_vector"], row["ndcg@10"] * 100), fontsize=7)
    left.axhline(report["recall_target"], ls="--", c="grey", lw=0.8)
    left.set_xscale("log")
    left.set_xlabel("single-query latency (ms, 1 thread)")
    left.set_ylabel("recall@10 vs exact")
    left.legend(fontsize=7)
    right.set_xscale("log")
    right.set_xlabel("bytes per vector")
    right.set_ylabel("nDCG@10")
    fig.suptitle(f"FAISS indexes on {report['dataset']} ({report['docs']:,} docs, d={report['dim']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
