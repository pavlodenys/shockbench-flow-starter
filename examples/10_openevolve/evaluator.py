"""Read literal parameters and evaluate a frozen agent on fixed training episodes."""

import ast
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


BOUNDS = {"reserve_weeks": (0.0, 8.0), "discount": (0.5, 1.0), "allocation_blend": (0.0, 1.0)}
ADAPTIVE_BOUNDS = {
    **BOUNDS,
    "delay_weight": (-2.0, 2.0),
    "spread_weight": (-2.0, 2.0),
    "shortage_weight": (-2.0, 2.0),
    "reserve_min": (0.0, 8.0),
    "reserve_max": (0.0, 12.0),
    "lp_shortage_weight": (-1.0, 1.0),
    "lp_delay_weight": (-1.0, 1.0),
    "lng_reserve_weight": (-4.0, 4.0),
    "lng_ration_weight": (0.0, 2.0),
    "lng_terminal_weight": (0.0, 1.0),
    "lng_lp_weight": (-1.0, 1.0),
    "semi_allocation_weight": (0.0, 1.0),
    "semi_shortage_weight": (-2.0, 2.0),
    "semi_value_weight": (-2.0, 2.0),
    "semi_delay_weight": (-2.0, 2.0),
}

MPC_BOUNDS = {
    **BOUNDS,
    "planning_horizon": (8.0, 32.0),
    "mpc_blend": (0.0, 1.0),
    "eta_margin": (0.0, 3.0),
    "supply_factor": (0.5, 1.2),
    "demand_factor": (0.8, 1.2),
    "terminal_weeks": (0.0, 4.0),
    "terminal_weight": (0.0, 0.5),
    "change_penalty": (0.0, 0.2),
}


def strategy_bounds(strategy):
    return {"constant": BOUNDS, "adaptive": ADAPTIVE_BOUNDS, "mpc": MPC_BOUNDS}[strategy]


def training_savings(result, baseline):
    """Mixed objective is the worst network, so Small cannot hide Full regression."""
    if "tasks" in result:
        values = {
            task: 100 * (1 - sum(rows["cost_cents"]) / sum(baseline["tasks"][task]["cost_cents"]))
            for task, rows in result["tasks"].items()
        }
        return min(values.values()), values
    return 100 * (1 - sum(result["cost_cents"]) / sum(baseline["cost_cents"])), {}


def read_params(code, strategy="constant"):
    """Accept only a literal PARAMS mapping; never execute model-generated Python."""
    if strategy not in ("constant", "adaptive", "mpc"):
        raise ValueError("Unknown strategy")
    bounds = strategy_bounds(strategy)
    tree = ast.parse(code)
    nodes = tree.body
    if nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant):
        if isinstance(nodes[0].value.value, str):
            nodes = nodes[1:]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Assign):
        raise ValueError("Return only PARAMS = {...}; imports, functions and other statements are not allowed")
    assignment = nodes[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise ValueError("Expected one PARAMS assignment")
    if assignment.targets[0].id != "PARAMS":
        raise ValueError("Expected PARAMS")
    values = ast.literal_eval(assignment.value)
    if not isinstance(values, dict) or set(values) != set(bounds):
        raise ValueError(f"Use exactly these keys: {list(bounds)}")
    params = {}
    for key, (low, high) in bounds.items():
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a finite number between {low} and {high}")
        params[key] = float(value)
    if strategy == "adaptive" and params["reserve_min"] > params["reserve_max"]:
        raise ValueError("reserve_min must not exceed reserve_max")
    return params


def candidate_id(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def write_candidate(run, params):
    folder = run / "candidates" / candidate_id(params)
    folder.mkdir(parents=True, exist_ok=True)
    source = run / "candidate-source" / "agent.py"
    if not source.exists():
        source = run / "baseline" / "agent.py"
    (folder / "agent.py").write_bytes(source.read_bytes())
    (folder / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    return folder


def summarize(rows):
    return {
        "cost_cents": [int(r["J_policy_cents"]) for r in rows],
        "fallback_weeks": sum(int(r["fallback_weeks"]) for r in rows),
        "cpu_weeks": sum(int(r["cpu_weeks"]) for r in rows),
        "invalid_entries": sum(int(r["invalid_entries"]) for r in rows),
    }


def play_candidate(folder, manifest):
    # OpenEvolve forks its workers. Never initialize/inherit joblib's pool in those
    # workers: each benchmark gets a fresh interpreter and its own episode workers.
    with tempfile.TemporaryDirectory(prefix="sbf-evolve-eval-") as tmp:
        request = Path(tmp) / "request.json"
        result = Path(tmp) / "result.json"
        request.write_text(json.dumps(manifest))
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(Path(__file__).with_name("play.py")),
                str(folder),
                str(request),
                str(result),
            ],
            check=True,
            timeout=3600,
        )
        return json.loads(result.read_text())


def evaluate(program_path):
    from openevolve.evaluation_result import EvaluationResult

    run = Path(os.environ["SBF_EVOLVE_RUN"])
    manifest = json.loads((run / "manifest.json").read_text())
    try:
        params = read_params(Path(program_path).read_text(), manifest.get("strategy", "constant"))
    except (ValueError, SyntaxError, TypeError) as exc:
        return EvaluationResult(metrics={"combined_score": -1e6}, artifacts={"error": str(exc)})
    folder = write_candidate(run, params)
    result_file = folder / "evaluation.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
    else:
        print(f"Evaluating {params} on {manifest['episodes']} {manifest['task']} training episodes", flush=True)
        result = play_candidate(folder, manifest)
        result_file.write_text(json.dumps(result, indent=2))
    baseline = json.loads((run / "baseline-evaluation.json").read_text())
    issues = result["fallback_weeks"] + result["cpu_weeks"] + result["invalid_entries"]
    savings, task_savings = training_savings(result, baseline)
    metrics = {
        "combined_score": -1e6 if issues else savings,
        "training_cost_saving_percent": savings,
        **{f"{t}_cost_saving_percent": v for t, v in task_savings.items()},
    }
    extra = [(a - b) / 100 for a, b in zip(result["cost_cents"], baseline["cost_cents"], strict=True)]
    artifacts = {
        "params": params,
        "training": {**result, "worst_extra_cost_usd": max(extra), "wins": sum(v < 0 for v in extra)},
        "note": "Training cost saving vs frozen mine, NOT official RSS or independent validation.",
    }
    print(f"Training cost saving: {savings:+.4f}%; fallback/CPU/invalid: {issues}", flush=True)
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
