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
from .router import MarginRouter

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
    searcher = HybridSearcher.open(dataset, cfg)
    ds = searcher.ds
    qids = list(ds.queries)
    texts = [ds.queries[q] for q in qids]

    started = time.perf_counter()
    runs = searcher.run_batch(texts, progress=True)
    elapsed = time.perf_counter() - started

    systems, ndcg = {}, {}
    for name, results in runs.items():
        if name == "margins":
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
            "margins": runs["margins"],
            "ndcg": {name: [vals[q] for q in qids] for name, vals in ndcg.items()},
        },
    )
    return report


TUNE_SPLITS = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}


def tune(splits=None, bm25_grid=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0), rerank_grid=(0.0, 0.5, 1.0, 1.5, 2.0), cfg=None):
    from dataclasses import replace

    from .rerank import Reranker

    cfg = cfg or Config()
    splits = splits or TUNE_SPLITS
    reranker = Reranker(cfg.reranker_model, cfg.device)

    prepared = []
    for name, split in splits.items():
        searcher = HybridSearcher.open(name, cfg, with_reranker=False)
        held = data.load(name, split=split)
        qids = list(held.queries)
        texts = [held.queries[q] for q in qids]
        runs = searcher.run_batch(texts, rerank=False)
        prepared.append((name, searcher, held.qrels, qids, texts, runs))

    def ndcg(results, searcher, qrels, qids):
        run = metrics.to_run(results, qids, searcher.ds.doc_ids)
        return metrics.summarize(run, qrels)["ndcg@10"]

    grid = []
    for wb in bm25_grid:
        per_dataset = []
        for _, searcher, qrels, qids, texts, runs in prepared:
            searcher.cfg = replace(cfg, w_bm25=wb)
            fused = [searcher._fuse(b[0], d[0]) for b, d in zip(runs["bm25"], runs["dense"])]
            ce = reranker.rerank_many(texts, [ids[: cfg.rerank_depth] for ids, _ in fused], searcher.ds.texts)
            per_dataset.append((searcher, qrels, qids, runs, ce))
        for wr in rerank_grid:
            scores = {}
            for (name, *_), (searcher, qrels, qids, runs, ce) in zip(prepared, per_dataset):
                searcher.cfg = replace(cfg, w_bm25=wb, w_rerank=wr)
                final = [searcher._fuse(b[0], d[0], c[0]) for b, d, c in zip(runs["bm25"], runs["dense"], ce)]
                scores[name] = ndcg(final, searcher, qrels, qids)
            grid.append({"w_bm25": wb, "w_rerank": wr, "ndcg@10": scores, "mean": float(np.mean(list(scores.values())))})
            print(f"w_bm25={wb:<4} w_rerank={wr:<4} " + "  ".join(f"{k} {v * 100:.2f}" for k, v in scores.items())
                  + f"  mean {grid[-1]['mean'] * 100:.2f}", flush=True)

    dense_only = {name: ndcg(runs["dense"], searcher, qrels, qids) for name, searcher, qrels, qids, _, runs in prepared}
    best = max(grid, key=lambda g: g["mean"])
    report = {"splits": splits, "dense_only": dense_only, "grid": grid, "best": best, "config": cfg.to_dict()}
    write_json(RESULTS_DIR / "tune.json", report)
    return report


def _latency(searcher, queries, mode, warmup=5):
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


def _recovered(base, full, routed):
    gain = full.mean() - base.mean()
    return None if gain <= 0 else float((routed - base.mean()) / gain)


def routing(dataset, cfg=None, recover=0.9, seed=7, latency_queries=200, out_dir=None):
    cfg = cfg or Config()
    out_dir = out_dir or RESULTS_DIR / "routing"
    per_query_file = RESULTS_DIR / "eval" / f"{dataset}.per_query.json"
    if not per_query_file.exists():
        evaluate(dataset, cfg)
    pq = json.loads(per_query_file.read_text())

    qids = pq["query_ids"]
    margins = np.array(pq["margins"])
    fused = np.array(pq["ndcg"]["hybrid"])
    full = np.array(pq["ndcg"]["hybrid+rerank"])

    order = list(range(len(qids)))
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    calib, held = np.array(sorted(order[:half])), np.array(sorted(order[half:]))

    router = MarginRouter.calibrate(margins[calib], fused[calib], full[calib], recover)
    sent = margins[held] < router.threshold
    routed_ndcg = float(np.where(sent, full[held], fused[held]).mean())
    share = float(sent.mean())
    random_ndcg = float(fused[held].mean() + share * (full[held].mean() - fused[held].mean()))

    router.save(artifact_dir(dataset, cfg) / "router.json")

    searcher = HybridSearcher.open(dataset, cfg)
    searcher.router = router
    rng = random.Random(seed)
    held_ids = [qids[i] for i in held]
    sample = rng.sample(held_ids, min(latency_queries, len(held_ids)))
    texts = [searcher.ds.queries[q] for q in sample]

    report = {
        "dataset": dataset,
        "recover_target": recover,
        "threshold": router.threshold,
        "calibration_queries": int(len(calib)),
        "heldout_queries": int(len(held)),
        "heldout": {
            "ndcg_hybrid": float(fused[held].mean()),
            "ndcg_full_rerank": float(full[held].mean()),
            "ndcg_routed": routed_ndcg,
            "ndcg_random_same_share": random_ndcg,
            "retained": routed_ndcg / float(full[held].mean()),
            "gain_recovered": _recovered(fused[held], full[held], routed_ndcg),
            "reranked_share": share,
        },
        "latency": {
            "device": searcher.reranker.device,
            "queries": len(texts),
            "never": _latency(searcher, texts, "never"),
            "always": _latency(searcher, texts, "always"),
            "auto": _latency(searcher, texts, "auto"),
        },
        "env": environment(),
    }
    lat = report["latency"]
    report["latency"]["mean_cut"] = 1 - lat["auto"]["mean_ms"] / lat["always"]["mean_ms"]
    write_json(out_dir / f"{dataset}.json", report)
    return report


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
