"""Local Ollama/OpenEvolve parameter search; run through scripts/openevolve-docker.ps1."""

import ast
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

import fire
from evaluator import (
    candidate_id,
    evaluate,
    play_candidate,
    read_params,
    strategy_bounds,
    write_candidate,
)
from proposal_model import create_proposal_client, proposal_stats


HERE = Path(__file__).resolve().parent


def main(
    task="small",
    episodes=8,
    entropy=202610051,
    iterations=10,
    n_jobs=2,
    model="qwen2.5-coder:3b",
    ollama="http://host.docker.internal:11434",
    out=None,
    strategy="constant",
    seed=None,
    warmup=False,
    temperature=0.95,
    proposal_attempts=4,
    min_change=0.03,
    context=8192,
):
    from openevolve import OpenEvolve
    from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, LLMModelConfig, PromptConfig

    if task not in ("tiny", "small", "full", "mixed") or min(episodes, iterations, n_jobs) < 1 or entropy <= 0:
        raise ValueError("Use tiny/small/full, positive counts, and a fresh nonzero training entropy")
    if strategy not in ("constant", "adaptive", "mpc"):
        raise ValueError("Use --strategy=constant, adaptive or mpc")
    if not 0 < temperature <= 1.2 or not 0 < min_change <= 1 or proposal_attempts < 1 or context < 2048:
        raise ValueError(
            "Use temperature in (0,1.2], min_change in (0,1], positive proposal_attempts and context >=2048"
        )
    tags = json.load(urllib.request.urlopen(ollama.rstrip("/") + "/api/tags", timeout=10))
    if model not in {m["name"] for m in tags["models"]}:
        raise ValueError(f"Model {model} is not installed in Ollama")
    run = Path(out or f"outputs/10_openevolve/{time.strftime('%Y-%m-%d_%H-%M-%S')}").resolve()
    if run.exists():
        raise ValueError(f"Run directory already exists; choose a new --out: {run}")
    source = Path("agents/mine/agent.py").resolve()
    source_code = source.read_text()
    assignment = next(
        n
        for n in ast.parse(source_code).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in n.targets)
    )
    params = ast.literal_eval(assignment.value)
    if source.with_name("params.json").exists():
        params.update(json.loads(source.with_name("params.json").read_text()))
    baseline_params = dict(params)
    candidate_source = source
    if strategy in ("adaptive", "mpc"):
        candidate_source = Path(
            "agents/adaptive_reserve/agent.py" if strategy == "adaptive" else "agents/mpc/agent.py"
        ).resolve()
        params = json.loads(candidate_source.with_name("params.json").read_text())
        # Keep the zero-weight seed anchored to the currently frozen mine.
        params.update(baseline_params)
    if seed is not None:
        seed_path = Path(seed)
        if seed_path.is_dir():
            seed_path = seed_path / "params.json"
        params.update(json.loads(seed_path.read_text()))
    seed_code = (
        f'"""Fuel planning parameters for the frozen {strategy} candidate."""\n\n'
        f"# EVOLVE-BLOCK-START\nPARAMS = {params!r}\n# EVOLVE-BLOCK-END\n"
    )
    params = read_params(seed_code, strategy)
    run.mkdir(parents=True)
    (run / "baseline").mkdir()
    (run / "baseline" / "agent.py").write_bytes(source.read_bytes())
    (run / "baseline" / "params.json").write_text(json.dumps(baseline_params, indent=2) + "\n")
    (run / "candidate-source").mkdir()
    (run / "candidate-source" / "agent.py").write_bytes(candidate_source.read_bytes())
    (run / "seed.py").write_text(seed_code)
    manifest = {
        "task": task,
        "strategy": strategy,
        "baseline_params": baseline_params,
        "seed_params": params,
        "seed": str(seed) if seed is not None else None,
        "warmup": bool(warmup),
        "proposal_driver": "ollama_numeric_patch",
        "proposal_config": {
            "temperature": temperature,
            "top_p": 0.95,
            "attempts": proposal_attempts,
            "min_change": min_change,
            "context": context,
            "seed": 20261005,
        },
        "episodes": episodes,
        "entropy": entropy,
        "iterations": iterations,
        "n_jobs": n_jobs,
        "model": model,
        "ollama": ollama,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(candidate_source.read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("openevolve", "shockbench-flow", "numpy", "scipy")},
        "objective": "Worst Small/Full cost saving percent vs frozen mine; NOT official RSS"
        if task == "mixed"
        else "Actual summed training cost saving percent vs frozen mine; NOT official RSS",
        "independent_validation": False,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Run: {run}\nPreparing frozen mine baseline on training root {entropy}", flush=True)
    baseline = play_candidate(run / "baseline", manifest)
    if baseline["fallback_weeks"] or baseline["cpu_weeks"] or baseline["invalid_entries"]:
        raise RuntimeError(f"Baseline has invalid actions or fallback: {baseline}")
    (run / "baseline-evaluation.json").write_text(json.dumps(baseline, indent=2))
    seed_folder = write_candidate(run, params)
    seed_result = (
        baseline if strategy == "constant" and params == baseline_params else play_candidate(seed_folder, manifest)
    )
    (seed_folder / "evaluation.json").write_text(json.dumps(seed_result, indent=2))
    os.environ["SBF_EVOLVE_RUN"] = str(run)
    config = Config(
        language="python",
        max_iterations=iterations,
        checkpoint_interval=1,
        random_seed=20261005,
        diff_based_evolution=False,
        max_code_length=4000,
        llm=LLMConfig(
            api_base=ollama.rstrip("/") + "/v1",
            api_key="ollama",
            models=[LLMModelConfig(name=model, weight=1.0)],
            temperature=temperature,
            top_p=0.95,
            max_tokens=1200 if strategy == "adaptive" else 300,
            timeout=300,
            retries=1,
        ),
        prompt=PromptConfig(
            system_message=(
                "Optimize fuel planning parameters in a fixed supply-network agent. "
                "Return Python containing only PARAMS = {...}, with exactly these numeric keys and bounds: "
                + json.dumps(strategy_bounds(strategy))
                + ". reserve_min must not exceed reserve_max when present. "
                "Higher training_cost_saving_percent is better; baseline is zero. "
                "Reserve controls target fuel coverage in weeks; discount controls future shortage penalties; "
                "allocation_blend controls the share of the LP allocation versus nominal shipments. "
                "For adaptive mode: reserve = clip(reserve_weeks + delay_weight*delay + "
                "spread_weight*spread + shortage_weight*shortage, reserve_min, reserve_max). "
                "Signals are weeks, capped at 12. Delay is excess ETA over nominal travel on the least "
                "delayed available route; spread is max ETA minus min ETA; shortage is the fuel coverage "
                "gap before earliest arrival. Negative weights reduce reserve. "
                "Adaptive LP share is allocation_blend + lp_shortage_weight*min(shortage/4,1) "
                "+ lp_delay_weight*min(delay/4,1), clipped to [0,1]. "
                "LNG only: lng_reserve_weight adds reserve weeks; lng_lp_weight adds LP share; "
                "lng_ration_weight multiplies the grid's rationing stock floor; "
                "lng_terminal_weight blends pooled inventory with a terminal-to-grid delivery forecast. "
                "Semiconductors: semi_allocation_weight blends template requests with stock/edge-budgeted "
                "allocation; semi_shortage_weight favors unmet factory or market needs; "
                "semi_value_weight favors economic value; semi_delay_weight favors shorter transport. "
                "With all extension weights zero, behavior equals frozen mine. "
                "Change at least one value. No imports, calls, functions or extra statements. "
                "Preserve the EVOLVE-BLOCK markers. Return the entire short program."
            ),
            num_top_programs=1,
            num_diverse_programs=1 if strategy == "adaptive" else 0,
            include_artifacts=True,
            max_artifact_bytes=4000 if strategy == "adaptive" else 1500,
        ),
        database=DatabaseConfig(population_size=20, archive_size=5, num_islands=1, log_prompts=True),
        evaluator=EvaluatorConfig(
            timeout=3600,
            max_retries=0,
            parallel_evaluations=1,
            cascade_evaluation=False,
            use_llm_feedback=False,
        ),
    )
    config.to_yaml(str(run / "config.yaml"))
    # Supported OpenEvolve factory hook. Attach after saving portable YAML;
    # the process controller pickles the top-level factory into its worker.
    for model_config in config.llm.models:
        model_config.init_client = create_proposal_client
    search = OpenEvolve(
        str(run / "seed.py"), str(HERE / "evaluator.py"), config=config, output_dir=str(run / "evolution")
    )
    if strategy == "adaptive" and warmup:
        from openevolve.database import Program

        # Small local models may repeat a zero-weight seed. Give the population
        # measured alternatives, without editing or executing model-generated code.
        profiles = [
            params,
            {**params, "delay_weight": 0.5, "spread_weight": 0.25, "shortage_weight": 0.0},
            {**params, "delay_weight": 0.0, "spread_weight": 0.0, "shortage_weight": 0.5},
            {**params, "delay_weight": -0.25, "spread_weight": -0.25, "shortage_weight": 0.0},
            {**params, "lp_shortage_weight": 0.25, "lp_delay_weight": -0.1},
            {**params, "lng_terminal_weight": 0.5, "lng_ration_weight": 0.5},
            {
                **params,
                "semi_allocation_weight": 0.5,
                "semi_shortage_weight": 0.5,
                "semi_value_weight": 0.25,
                "semi_delay_weight": 0.25,
            },
        ]
        warmup = []
        for index, profile in enumerate(profiles):
            program_file = run / f"warmup-{index}.py"
            code = f"# EVOLVE-BLOCK-START\nPARAMS = {profile!r}\n# EVOLVE-BLOCK-END\n"
            program_file.write_text(code)
            result = evaluate(str(program_file))
            program = Program(
                id=str(uuid.uuid4()),
                code=code,
                language="python",
                changes_description=f"Measured adaptive reserve starting profile {index}",
                metrics=result.metrics,
            )
            search.database.add(program)
            search.database.store_artifacts(program.id, result.artifacts)
            warmup.append({"params": profile, "metrics": result.metrics})
        (run / "warmup.json").write_text(json.dumps(warmup, indent=2))
    best = asyncio.run(search.run(iterations=iterations))
    if best is None:
        raise RuntimeError("OpenEvolve returned no best program")
    best_code = best.code
    best_params = read_params(best_code, strategy)
    (run / "best-program.py").write_text(best_code)
    # A candidate export is reviewable; it is not automatically installed as mine.
    best_folder = write_candidate(run, best_params)
    (run / "best-candidate").mkdir()
    for filename in ("agent.py", "params.json"):
        shutil.copy2(best_folder / filename, run / "best-candidate" / filename)
    result = evaluate(str(run / "best-program.py"))
    summary = {
        "best_params": best_params,
        "metrics": result.metrics,
        "unique_evaluated_candidates": len(list((run / "candidates").glob("*/evaluation.json"))),
        "baseline_candidate_id": candidate_id(baseline_params),
        "seed_candidate_id": candidate_id(params),
        "best_candidate_id": candidate_id(best_params),
        "independent_validation": False,
        "promoted": False,
        "proposal_stats": proposal_stats(run),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Candidate: {run / 'best-candidate'}; next: independent Small/Full RSS comparisons", flush=True)
    if summary["proposal_stats"]["accepted_model_proposals"] == 0:
        raise RuntimeError("No new Qwen proposal was accepted; this run did not search. See proposals.jsonl")


if __name__ == "__main__":
    fire.Fire(main)
