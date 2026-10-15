import argparse
import os
import sys

from .config import Config

DATASETS = ("scifact", "nfcorpus", "fiqa")


def _config(args):
    overrides = {
        k: v
        for k, v in {
            "dense_model": getattr(args, "model", None),
            "reranker_model": getattr(args, "reranker", None),
            "index_factory": getattr(args, "index", None),
            "device": getattr(args, "device", None),
        }.items()
        if v is not None
    }
    return Config(**overrides)


def cmd_download(args):
    from . import data

    for name in args.datasets:
        path = data.download(data.parse_name(name)[0])
        print(f"{name}: {path}")


def cmd_index(args):
    from . import data
    from .pipeline import build_artifacts

    cfg = _config(args)
    for name in args.datasets:
        ds = data.load(name)
        build_artifacts(ds, cfg)
        print(f"{name}: {len(ds):,} docs indexed with {cfg.dense_model} / {cfg.index_factory}")


def cmd_search(args):
    from .pipeline import HybridSearcher

    searcher = HybridSearcher.open(args.dataset, _config(args), with_reranker=args.rerank != "never")
    res = searcher.search(args.query, k=args.k, rerank=args.rerank)
    print(f"reranked={res.reranked} margin={res.margin:.3f} timings={res.timings_ms}")
    for rank, hit in enumerate(res.hits, 1):
        print(f"{rank:2d}. [{hit.doc_id}] {hit.score:.4f}  {hit.text[:110]}")


def cmd_eval(args):
    from .experiments import evaluate

    cfg = _config(args)
    for name in args.datasets:
        report = evaluate(name, cfg)
        print(f"\n{name} ({report['docs']:,} docs, {report['queries']} queries)")
        for system, scores in report["systems"].items():
            row = "  ".join(f"{m} {v * 100:6.2f}" for m, v in scores.items())
            print(f"  {system:15s} {row}")


def cmd_tune(args):
    from .experiments import tune

    best = tune(cfg=_config(args))["best"]
    print(f"best on dev splits: w_bm25={best['w_bm25']} w_rerank={best['w_rerank']} (mean nDCG@10 {best['mean'] * 100:.2f})")


def cmd_bench(args):
    from .experiments import bench

    bench(args.dataset, _config(args), target=args.target, latency_queries=args.latency_queries)


def cmd_route(args):
    from .experiments import routing

    cfg = _config(args)
    for name in args.datasets:
        r = routing(name, cfg, recover=args.recover)
        h, lat = r["heldout"], r["latency"]
        print(
            f"{name}: rerank {h['reranked_share']:.0%} of queries, nDCG {h['ndcg_routed'] * 100:.2f} "
            f"(full rerank {h['ndcg_full_rerank'] * 100:.2f}, no rerank {h['ndcg_hybrid'] * 100:.2f}), "
            f"mean latency {lat['always']['mean_ms']:.0f} -> {lat['auto']['mean_ms']:.0f} ms ({lat['mean_cut']:.0%} less)"
        )


def cmd_report(args):
    from .report import build

    print(build())


def cmd_gate(args):
    from .gate import run_gate

    ok, lines = run_gate(args.dataset, args.tolerance, update=args.update, device=args.device or "cpu")
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"## Retrieval quality gate ({args.dataset})\n\n{text}\n")
    if not ok:
        print(f"\nquality dropped by more than {args.tolerance} points", file=sys.stderr)
        sys.exit(1)


def cmd_serve(args):
    import uvicorn

    from .api import create_app
    from .pipeline import HybridSearcher

    searcher = HybridSearcher.open(args.dataset, _config(args))
    uvicorn.run(create_app(searcher), host=args.host, port=args.port)


def cmd_fix_openmp(args):
    from . import openmp

    if openmp.fix():
        print("faiss now shares torch's libomp")
    else:
        print("nothing to fix")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hse", description="hybrid BM25 + dense search")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    def models(sp):
        sp.add_argument("--model")
        sp.add_argument("--reranker")
        sp.add_argument("--index")
        sp.add_argument("--device", choices=["cpu", "mps", "cuda"])

    sp = add("download", cmd_download, "fetch BEIR datasets")
    sp.add_argument("datasets", nargs="*", default=list(DATASETS))

    sp = add("index", cmd_index, "build BM25 + FAISS indexes")
    sp.add_argument("datasets", nargs="*", default=list(DATASETS))
    models(sp)

    sp = add("search", cmd_search, "run one query")
    sp.add_argument("query")
    sp.add_argument("--dataset", default="scifact")
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--rerank", choices=["auto", "always", "never"], default="auto")
    models(sp)

    sp = add("eval", cmd_eval, "nDCG / recall for every system")
    sp.add_argument("datasets", nargs="*", default=list(DATASETS))
    models(sp)

    sp = add("tune", cmd_tune, "grid-search fusion weights on dev/train splits")
    models(sp)

    sp = add("bench", cmd_bench, "FAISS index benchmark")
    sp.add_argument("--dataset", default="quora-200k")
    sp.add_argument("--target", type=float, default=0.95)
    sp.add_argument("--latency-queries", type=int, default=1000)
    models(sp)

    sp = add("route", cmd_route, "calibrate and measure margin-based rerank routing")
    sp.add_argument("datasets", nargs="*", default=list(DATASETS))
    sp.add_argument("--recover", type=float, default=0.9)
    models(sp)

    add("report", cmd_report, "write results/RESULTS.md from saved runs")

    sp = add("gate", cmd_gate, "fail if retrieval quality regressed")
    sp.add_argument("--dataset", default="scifact")
    sp.add_argument("--tolerance", type=float, default=0.5)
    sp.add_argument("--update", action="store_true")
    sp.add_argument("--device", choices=["cpu", "mps", "cuda"])

    sp = add("serve", cmd_serve, "start the FastAPI server")
    sp.add_argument("--dataset", default="scifact")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    models(sp)

    add("fix-openmp", cmd_fix_openmp, "macOS: make faiss use torch's OpenMP runtime")

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
