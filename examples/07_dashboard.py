"""Compare mine and send-the-maximum on the same dev scenario, with the naive rule.

    uv run python examples/07_dashboard.py
    uv run python examples/07_dashboard.py --episode=29 --quick
    uv run python examples/07_dashboard.py --task=small --episode=0

Writes network.png, dashboard_*.png, episode_*.gif and record_*.npz. The naive rule takes about 30 s on Tiny the
first time, then comes from the cache. Made for Tiny: on Small and Full the map and stock panels are crowded.
"""

import time
from pathlib import Path

import fire
import gymnasium as gym
from shockbench_flow_agent import NAIVE_REPLICATIONS, QUICK
from shockbench_flow_gym import dashboard  # also registers the ShockBench/* environments

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


def main(
    task: str = "tiny",
    episode: int | None = None,
    search: int = 60,
    min_open: float = 0.5,
    naive: bool = True,
    quick: bool = False,
    n_jobs: int = -1,
    regime: str = "standard",
    seed: int = 0,
    out: str | None = None,
) -> None:
    """Write the map, both dashboards, both GIFs and the records.

    Args:
        task: tiny, small or full.
        episode: a dev episode (default: the first in which a strait closes below ``min_open`` open).
        search: look through dev episodes 0 .. search - 1.
        min_open: the open fraction a strait must fall below.
        naive: also play the naive rule (``--nonaive`` to skip it).
        quick: a rough naive rule, not the board's.
        n_jobs: workers of the naive rule's first computation (-1: all cores).
        regime: the information regime; standard is the scored one.
        seed: the reset's seed.
        out: the run folder (default: outputs/07_dashboard/<date_time>).

    """
    out = Path(out or f"outputs/07_dashboard/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id(task), regime=regime)
    if episode is None:
        (n,) = episodes_with_closure(env, 1, search=search, min_open=min_open, seed=seed)
        print(f"dev episode {n}: a strait closes below {min_open:.0%} open")
    else:
        n = int(episode)
        print(f"dev episode {n}")
    replications = None if not naive else QUICK["fq_replications"] if quick else NAIVE_REPLICATIONS
    dashboard.plot_network(task).savefig(out / "network.png", dpi=100)
    written = ["network.png"]
    costs = {}
    for name, agent in (("mine", load("mine")), ("max", load("template"))):
        start = time.perf_counter()
        rec = dashboard.record_episode(
            env, agent, seed=seed, options={"episode": n}, naive_replications=replications, n_jobs=n_jobs
        )
        meta = rec["meta"]  # costs in integer cents
        costs[name] = meta["J_cents"] / 100
        naive_cost = f", the naive rule's ${meta['naive_J_cents'] / 100:,.0f}" if "naive" in rec else ""
        print(f"{name}: cost ${meta['J_cents'] / 100:,.0f}{naive_cost} ({time.perf_counter() - start:.1f} s)")
        dashboard.save_record(rec, out / f"record_{name}.npz")
        dashboard.episode_dashboard(rec, out / f"dashboard_{name}.png")
        dashboard.episode_animation(rec, out / f"episode_{name}.gif")
        written += [f"record_{name}.npz", f"dashboard_{name}.png", f"episode_{name}.gif"]
    difference = costs["mine"] - costs["max"]
    percent = f" ({100 * difference / costs['max']:+.2f}%)" if costs["max"] > 0 else ""
    print(f"mine minus template: ${difference:+,.0f}{percent}; negative means mine costs less")
    print(f"written in {out}: {', '.join(written)}")


if __name__ == "__main__":
    fire.Fire(main)
