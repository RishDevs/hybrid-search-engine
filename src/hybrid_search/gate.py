import json

from .config import RESULTS_DIR, Config
from .experiments import evaluate

BASELINE = RESULTS_DIR / "baseline.json"
GATED = ("dense", "hybrid", "hybrid+rerank")
METRICS = ("ndcg@10", "recall@100")


def run_gate(dataset="scifact", tolerance=0.5, baseline=BASELINE, update=False, device="cpu"):
    cfg = Config(device=device)
    report = evaluate(dataset, cfg, out_dir=RESULTS_DIR / "gate")
    current = {s: {m: report["systems"][s][m] for m in METRICS} for s in GATED}

    if update or not baseline.exists():
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(json.dumps({"dataset": dataset, "config": report["config"], "systems": current}, indent=2))
        return True, [f"baseline written to {baseline}"]

    saved = json.loads(baseline.read_text())
    lines, ok = [], True
    lines.append("| system | metric | baseline | current | delta (pts) |")
    lines.append("|---|---|---|---|---|")
    for system in GATED:
        for metric in METRICS:
            before = saved["systems"].get(system, {}).get(metric)
            now = current[system][metric]
            if before is None:
                lines.append(f"| {system} | {metric} | - | {now * 100:.2f} | new |")
                continue
            delta = (now - before) * 100
            failed = delta < -tolerance
            ok &= not failed
            mark = " FAIL" if failed else ""
            lines.append(f"| {system} | {metric} | {before * 100:.2f} | {now * 100:.2f} | {delta:+.2f}{mark} |")
    return ok, lines
