"""Read completed mixed evaluations without changing a running search."""

import json
import sys
from pathlib import Path


def analyze(folder):
    run = Path(folder)
    manifest = json.loads((run / "manifest.json").read_text())
    baseline = json.loads((run / "baseline-evaluation.json").read_text())
    candidates = []
    for file in sorted((run / "candidates").glob("*/evaluation.json")):
        result = json.loads(file.read_text())
        networks = {}
        for task, own in result["tasks"].items():
            old = baseline["tasks"][task]["cost_cents"]
            costs = own["cost_cents"]
            if len(costs) != len(old):
                raise ValueError("Mismatched episode counts")
            extra = [100 * (a / b - 1) for a, b in zip(costs, old, strict=True)]
            networks[task] = {
                "saving_percent": 100 * (1 - sum(costs) / sum(old)),
                "wins": sum(a < b for a, b in zip(costs, old, strict=True)),
                "losses": sum(a > b for a, b in zip(costs, old, strict=True)),
                "worst_episode": max(range(len(extra)), key=lambda i: extra[i]),
                "worst_extra_cost_percent": max(extra),
                "max_cpu_s": own["max_cpu_s"],
                "mpc_failures": own["mpc_failures"],
            }
        issues = result["cpu_weeks"] + result["invalid_entries"] + result["fallback_weeks"]
        candidates.append(
            {
                "id": file.parent.name,
                "params": json.loads((file.parent / "params.json").read_text()),
                "fitness": -1e6 if issues else min(n["saving_percent"] for n in networks.values()),
                "issues": issues,
                "networks": networks,
            }
        )
    candidates.sort(key=lambda c: -c["fitness"])
    events_file = run / "proposals.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    summary = {
        "completed": (run / "summary.json").exists(),
        "training_only": True,
        "objective": manifest["objective"],
        "entropy_small": manifest["entropy"],
        "entropy_full": manifest["entropy"] + 1,
        "episodes_per_network": manifest["episodes"],
        "evaluated_candidates": len(candidates),
        "accepted_proposals": sum(e["status"] == "accepted" for e in events),
        "exhausted_proposals": sum(e["status"] == "exhausted" for e in events),
        "rejected_proposals": sum(e["status"] == "rejected" for e in events),
        "improves_both_networks": bool(candidates and candidates[0]["fitness"] > 0),
        "candidates": candidates,
    }
    output = run / "analysis-training.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2))
    if candidates:
        print(json.dumps(candidates[0], indent=2))
    return summary


if __name__ == "__main__":
    analyze(sys.argv[1])
