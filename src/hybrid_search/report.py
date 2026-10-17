import json

from .config import RESULTS_DIR


def _load(folder):
    out = {}
    for f in sorted((RESULTS_DIR / folder).glob("*.json")):
        if f.name.endswith(".per_query.json"):
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def _pts(x):
    return f"{x * 100:.2f}"


def quality_section(evals):
    if not evals:
        return []
    systems = list(next(iter(evals.values()))["systems"])
    lines = ["## Retrieval quality (nDCG@10, BEIR test split)", ""]
    lines.append("| system | " + " | ".join(evals) + " | mean |")
    lines.append("|---|" + "---|" * (len(evals) + 1))
    means = {}
    for s in systems:
        vals = [e["systems"][s]["ndcg@10"] for e in evals.values() if s in e["systems"]]
        if len(vals) != len(evals):
            continue
        means[s] = sum(vals) / len(vals)
        lines.append(f"| {s} | " + " | ".join(_pts(v) for v in vals) + f" | {_pts(means[s])} |")
    lines.append("")
    if {"hybrid+rerank", "dense", "bm25"} <= means.keys():
        lines.append(
            f"Full pipeline vs dense-only: {(means['hybrid+rerank'] - means['dense']) * 100:+.2f} points; "
            f"vs BM25-only: {(means['hybrid+rerank'] - means['bm25']) * 100:+.2f} points (mean over {len(evals)} datasets)."
        )
        lines.append("")
    lines.append("Paired t-test p-values per dataset (significance differs by comparison and dataset):")
    lines.append("")
    for name, e in evals.items():
        ps = ", ".join(f"{k}: {v:.3g}" for k, v in e["p_values"].items())
        lines.append(f"- {name}: {ps}")
    lines.append("")
    return lines


def _quality(row):
    q = row.get("quality_point")
    if not q:
        return "not reached | -"
    param = f"{row['knob']}={q['param']}" if row["knob"] else "-"
    return f"{param} | {q['speedup_vs_flat']:.1f}x"


def bench_section(benches):
    lines = []
    for name, b in benches.items():
        lines += [
            f"## FAISS benchmark: {name} ({b['docs']:,} docs, {b['queries']:,} queries, d={b['dim']})",
            "",
            f"Latency is single-query, single-thread, averaged over {b['latency_queries']} queries. "
            f"Recall point = fastest setting with recall@10 >= {b['recall_target']} against exact search "
            "(or the best recall reached if none get there). "
            f"Quality point = fastest setting within {b['max_ndcg_loss_points']} nDCG points of exact search.",
            "",
            "| index | recall point | B/vector | compression | build s | recall@10 | latency ms | speedup | nDCG@10 "
            "| delta pts | quality point | speedup at quality point |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in b["indexes"]:
            op = r["operating_point"]
            param = f"{r['knob']}={op['param']}" if r["knob"] else "-"
            lines.append(
                f"| {r['index']} | {param} | {r['bytes_per_vector']:.0f} | {r['compression']:.1f}x | {r['build_s']:.1f} | "
                f"{op['recall@10']:.3f} | {op['latency_ms']:.3f} | {r['speedup_vs_flat']:.1f}x | "
                f"{_pts(r['ndcg@10'])} | {r['ndcg_delta_points']:+.2f} | {_quality(r)} |"
            )
        lines += [
            "",
            "Compression = float32 vectors ÷ serialized index size, i.e. the index held in memory for search.",
        ]
        rescored = [r for r in b["indexes"] if "rescored" in r]
        if rescored:
            lines += [
                "",
                "With top-100 rescoring against float32 vectors kept on disk. These rows keep the in-memory "
                "compression above, but total storage also includes the float32 vectors "
                f"({b['docs'] * b['dim'] * 4 / 1e6:.0f} MB here).",
                "",
            ]
            lines += ["| index | recall@10 | latency ms | nDCG@10 | delta pts |", "|---|---|---|---|---|"]
            for r in rescored:
                x = r["rescored"]
                lines.append(
                    f"| {r['index']} + rescore | {x['recall@10']:.3f} | {x['latency_ms']:.3f} | "
                    f"{_pts(x['ndcg@10'])} | {x['ndcg_delta_points']:+.2f} |"
                )
        lines += ["", f"![bench](bench/{name}.png)", ""]
    return lines


def routing_section(summary):
    if not summary:
        return []
    lines = [
        "## Learned rerank router (trained on dev splits, measured on test)",
        "",
        f"Ridge regression on {len(summary['weights']) - 1} query features predicts each query's reranker gain; "
        f"the threshold was set on cross-validated dev predictions to recover {summary['recover_target']:.0%} of the gain.",
        "",
        "| dataset | reranked | no rerank | always rerank | routed | random, same share | gain recovered |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, t in [*summary["test"].items(), ("all (pooled)", summary["test_pooled"])]:
        rec = "n/a" if t["gain_recovered"] is None else f"{t['gain_recovered']:.0%}"
        lines.append(
            f"| {name} | {t['reranked_share']:.0%} | {_pts(t['ndcg_hybrid'])} | {_pts(t['ndcg_full_rerank'])} | "
            f"{_pts(t['ndcg_routed'])} | {_pts(t['ndcg_random_same_share'])} | {rec} |"
        )
    lat = summary["latency_mean_ms"]
    lines += [
        "",
        f"Mean end-to-end latency per query (mean over datasets, `cpu`): always rerank {lat['always']:.0f} ms, "
        f"routed {lat['auto']:.0f} ms ({lat['cut']:.0%} less).",
        "",
        "Router weights (standardized features): "
        + ", ".join(f"{k} {v:+.4f}" for k, v in summary["weights"].items()),
        "",
    ]
    return lines


def build():
    evals, benches = _load("eval"), _load("bench")
    route_file = RESULTS_DIR / "routing" / "summary.json"
    routes = json.loads(route_file.read_text()) if route_file.exists() else None
    lines = ["# Results", "", "Generated by `hse report` from the JSON files in this folder.", ""]
    lines += quality_section(evals) + bench_section(benches) + routing_section(routes)
    env = next((x["env"] for x in [*evals.values(), *benches.values()]), None)
    if env:
        lines += ["## Environment", "", "```", json.dumps(env, indent=2), "```", ""]
    path = RESULTS_DIR / "RESULTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
