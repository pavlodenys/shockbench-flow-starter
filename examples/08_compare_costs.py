"""Compare episode costs on a private training root; no oracle computation is needed.

    uv run python examples/08_compare_costs.py --agents=[baseline,mine] --task=small

These costs diagnose policies; use sbf compare on held-out scenarios for the RSS report.
"""

import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load


def main(agents=("baseline", "mine"), task="tiny", episodes=8, entropy=12345, out=None):
    """Compare names or comma-separated paths on identical training scenarios."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if entropy == 0:
        raise ValueError("Use a separate training root; reserve root 0 for confirmation.")
    if isinstance(agents, str):
        agents = [name.strip() for name in agents.strip("[]").split(",")]
    out = Path(out or f"outputs/08_compare_costs/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in agents:
        cls = load(name)
        env = gym.make(env_id(task), entropy=entropy)
        try:
            for episode in range(episodes):
                obs, info = env.reset(seed=episode, options={"episode": episode})
                config = agent_config_from_reset(env, obs, info)
                start = time.process_time()
                agent = cls(config)
                init_cpu = time.process_time() - start
                total, max_cpu = 0.0, 0.0
                costs = np.zeros(len(config["layout"]["cost_components"]))
                for week in range(config["T"]):
                    start = time.process_time()
                    action = agent.act(obs)
                    cpu = time.process_time() - start + (init_cpu if week == 0 else 0)
                    max_cpu = max(max_cpu, cpu)
                    flows = np.asarray(action["flows"])
                    assert flows.shape == env.action_space["flows"].shape
                    assert np.all(np.isfinite(flows)) and np.all(flows >= 0)
                    obs, reward, done, truncated, info = env.step(action)
                    total -= float(reward)
                    costs += obs["last_week.cost_components"]
                    if done or truncated:
                        break
                row = dict(
                    agent=str(name),
                    task=task,
                    entropy=entropy,
                    episode=episode,
                    cost_usd=total,
                    salvage_usd=info.get("salvage_cents", 0) / 100,
                    solver_failures=getattr(agent, "failures", 0),
                    max_cpu_s=max_cpu,
                    components=dict(zip(config["layout"]["cost_components"], costs.tolist())),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                (out / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
        finally:
            env.close()
    for name in agents:
        own = [r["cost_usd"] for r in rows if r["agent"] == str(name)]
        print(f"{name}: mean cost {np.mean(own):,.0f} USD ({episodes} episodes, root {entropy})", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
