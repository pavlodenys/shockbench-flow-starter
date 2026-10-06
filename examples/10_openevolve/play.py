"""One benchmark evaluation in a clean process, separate from OpenEvolve's pool."""

import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
from evaluator import summarize
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.scoring import episode_set


def play_cost_episode(args):
    folder, task, episode, entropy = args
    env = gym.make(env_id(task), entropy=entropy)
    try:
        obs, info = env.reset(seed=episode, options={"episode": episode})
        start = time.process_time()
        agent = load(folder)(agent_config_from_reset(env, obs, info))
        init_cpu = time.process_time() - start
        total = 0.0
        cpu_weeks = invalid = 0
        max_cpu = 0.0
        for week in range(agent.T):
            start = time.process_time()
            action = agent.act(obs)
            elapsed = time.process_time() - start + (init_cpu if week == 0 else 0.0)
            max_cpu = max(max_cpu, elapsed)
            cpu_weeks += elapsed > (4.0 if task == "full" else 2.0)
            flows = np.asarray(action["flows"])
            valid = flows.shape == env.action_space["flows"].shape and np.all(np.isfinite(flows)) and np.all(flows >= 0)
            if not valid:
                invalid += 1
                return {
                    "cost_cents": 0,
                    "cpu_weeks": cpu_weeks,
                    "invalid_entries": invalid,
                    "max_cpu_s": max_cpu,
                    "mpc_failures": getattr(agent, "mpc_failures", 0),
                }
            obs, reward, done, truncated, info = env.step(action)
            total -= float(reward)
            if done or truncated:
                break
        return {
            "cost_cents": int(round(total * 100)),
            "cpu_weeks": cpu_weeks,
            "invalid_entries": invalid,
            "max_cpu_s": max_cpu,
            "mpc_failures": getattr(agent, "mpc_failures", 0),
            "mpc_solves": getattr(agent, "mpc_solves", 0),
        }
    finally:
        env.close()


def play_mixed(folder, manifest):
    # Direct policy costs: no oracle/reference computation during model search.
    jobs = [
        (folder, task, episode, manifest["entropy"] + index)
        for index, task in enumerate(("small", "full"))
        for episode in range(manifest["episodes"])
    ]
    with ProcessPoolExecutor(max_workers=manifest["n_jobs"], mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(play_cost_episode, jobs))
    results = {}
    for index, task in enumerate(("small", "full")):
        own = rows[index * manifest["episodes"] : (index + 1) * manifest["episodes"]]
        results[task] = {
            "cost_cents": [r["cost_cents"] for r in own],
            "fallback_weeks": 0,
            "cpu_weeks": sum(r["cpu_weeks"] for r in own),
            "invalid_entries": sum(r["invalid_entries"] for r in own),
            "max_cpu_s": max(r["max_cpu_s"] for r in own),
            "mpc_failures": sum(r["mpc_failures"] for r in own),
            "mpc_solves": sum(r.get("mpc_solves", 0) for r in own),
        }
    return {
        "tasks": results,
        "cost_cents": [r["cost_cents"] for r in rows],
        "fallback_weeks": 0,
        "cpu_weeks": sum(r["cpu_weeks"] for r in rows),
        "invalid_entries": sum(r["invalid_entries"] for r in rows),
        "cpu_check": "measured process time; isolated sbf check required before RSS validation",
    }


def main(folder, request, result):
    manifest = json.loads(Path(request).read_text())
    if manifest["task"] == "mixed":
        Path(result).write_text(json.dumps(play_mixed(folder, manifest)))
        return
    # Quick references only prepare episodes. Actual policy costs determine fitness;
    # independent official RSS validation is performed after search.
    es = episode_set(
        manifest["task"],
        manifest["episodes"],
        entropy=manifest["entropy"],
        quick=True,
        n_jobs=manifest["n_jobs"],
        verbose=False,
    )
    rows = es.play(folder, cpu_budget=True, n_jobs=manifest["n_jobs"])
    Path(result).write_text(json.dumps(summarize(rows)))


if __name__ == "__main__":
    main(*sys.argv[1:])
