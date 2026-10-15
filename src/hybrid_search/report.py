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
    lines.append("Paired t-test p-values per dataset:")
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
        rescored = [r for r in b["indexes"] if "rescored" in r]
        if rescored:
            lines += ["", "With top-100 rescoring against float32 vectors kept on disk:", ""]
            lines += ["| index | recall@10 | latency ms | nDCG@10 | delta pts |", "|---|---|---|---|---|"]
            for r in rescored:
                x = r["rescored"]
                lines.append(
                    f"| {r['index']} + rescore | {x['recall@10']:.3f} | {x['latency_ms']:.3f} | "
                    f"{_pts(x['ndcg@10'])} | {x['ndcg_delta_points']:+.2f} |"
                )
        lines += ["", f"![bench](bench/{name}.png)", ""]
    return lines


def routing_section(routes):
    if not routes:
        return []
    lines = [
        "## Margin-based rerank routing (held-out half of queries)",
        "",
        "| dataset | reranked | no rerank | full rerank | routed | random, same share | gain recovered "
        "| full rerank ms | routed ms | mean cut |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in routes.items():
        h, lat = r["heldout"], r["latency"]
        rec = "n/a (reranker hurts)" if h["gain_recovered"] is None else f"{h['gain_recovered']:.0%}"
        lines.append(
            f"| {name} | {h['reranked_share']:.0%} | {_pts(h['ndcg_hybrid'])} | {_pts(h['ndcg_full_rerank'])} | "
            f"{_pts(h['ndcg_routed'])} | {_pts(h['ndcg_random_same_share'])} | {rec} | "
            f"{lat['always']['mean_ms']:.0f} | {lat['auto']['mean_ms']:.0f} | {lat['mean_cut']:.0%} |"
        )
    dev = next(iter(routes.values()))["latency"]["device"]
    lines += ["", f"Latency measured end to end per query on `{dev}`.", ""]
    return lines


def build():
    evals, benches, routes = _load("eval"), _load("bench"), _load("routing")
    lines = ["# Results", "", "Generated by `hse report` from the JSON files in this folder.", ""]
    lines += quality_section(evals) + bench_section(benches) + routing_section(routes)
    env = next((x["env"] for x in [*evals.values(), *benches.values()]), None)
    if env:
        lines += ["## Environment", "", "```", json.dumps(env, indent=2), "```", ""]
    path = RESULTS_DIR / "RESULTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
