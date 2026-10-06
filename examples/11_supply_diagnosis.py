"""Diagnose fuel and chip losses on a private training root, without an oracle.

The agent receives only its public config and observation. Simulator records are
read after each action to explain energy, production and disposal losses.
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


def main(agent="mine", task="small", episodes=8, entropy=202610071, out=None):
    if entropy == 0 or episodes < 1:
        raise ValueError("Use a private training root and a positive episode count")
    out = Path(out or f"outputs/11_supply_diagnosis/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id(task), entropy=entropy)
    rows = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=episode, options={"episode": episode})
            config = agent_config_from_reset(env, obs, info)
            static, layout = config["static"], config["layout"]
            names = static["nodes"]["id"]
            goods = static["commodities"]["id"]
            slots = {tuple(pair): i for i, pair in enumerate(layout["stock_slots"])}
            record_positions = [env.unwrapped.instance.slot_index[tuple(pair)] for pair in layout["stock_slots"]]
            raw = static["instance"]["nodes"]
            fabs = layout["fabs"]
            policy = load(agent)(config)
            costs = np.zeros(len(layout["cost_components"]))
            production, capacity, input_gap, energy_gap = (np.zeros(len(fabs)) for _ in range(4))
            shed = np.zeros(len(layout["grids"]))
            shortage = np.zeros(len(layout["demands"]))
            disposed = np.zeros(len(layout["stock_slots"]))
            total, max_cpu = 0.0, 0.0
            weeks = []
            for _ in range(config["T"]):
                start = time.process_time()
                action = policy.act(obs)
                max_cpu = max(max_cpu, time.process_time() - start)
                cap = obs["graph_now.fab.cap_eff"].copy()
                available = []
                for node in fabs:
                    k = goods.index(raw[node]["fab"]["input"])
                    value = obs["stock.qty"][slots[node, k]]
                    arrived = (
                        (obs["pipeline.arrival_week"] == obs["week"][0])
                        & (obs["pipeline.arrival_week.observed"] == 1)
                        & (obs["pipeline.edge.observed"] == 1)
                        & (obs["pipeline.k.observed"] == 1)
                        & (obs["pipeline.qty.observed"] == 1)
                        & (obs["pipeline.k"] == k)
                    )
                    value += sum(
                        obs["pipeline.qty"][j]
                        for j in np.flatnonzero(arrived)
                        if static["edges"]["head"][int(obs["pipeline.edge"][j])] == node
                    )
                    available.append(value)
                feasible = np.minimum(cap, available)
                obs, reward, done, truncated, info = env.step(action)
                record = env.unwrapped.core.trajectory.records[-1]
                costs += obs["last_week.cost_components"]
                total -= float(reward)
                production += record.lots_started
                capacity += cap
                input_gap += np.maximum(0.0, cap - feasible)
                energy_gap += np.maximum(0.0, feasible - record.lots_started)
                shed += record.shed
                shortage += record.lost + record.backlog
                disposed += record.disposal[record_positions]
                weeks.append(
                    dict(
                        week=record.week,
                        costs=obs["last_week.cost_components"].tolist(),
                        lots_started=record.lots_started.tolist(),
                        fab_capacity=cap.tolist(),
                        input_gap=np.maximum(0.0, cap - feasible).tolist(),
                        energy_gap=np.maximum(0.0, feasible - record.lots_started).tolist(),
                        shed=record.shed.tolist(),
                        stock=obs["stock.qty"].tolist(),
                        disposal=record.disposal[record_positions].tolist(),
                        served=record.served.tolist(),
                        demand=record.demand.tolist(),
                        backlog=record.backlog.tolist(),
                    )
                )
                if done or truncated:
                    break
            row = dict(
                agent=agent,
                task=task,
                entropy=entropy,
                episode=episode,
                cost_usd=total,
                max_cpu_s=max_cpu,
                solver_failures=getattr(policy, "failures", 0),
                mpc_failures=getattr(policy, "mpc_failures", 0),
                chain_failures=getattr(policy, "chain_failures", 0),
                components=dict(zip(layout["cost_components"], costs.tolist())),
                fabs=[
                    dict(
                        node=names[node],
                        starts=production[i],
                        capacity=capacity[i],
                        input_gap=input_gap[i],
                        energy_gap=energy_gap[i],
                    )
                    for i, node in enumerate(fabs)
                ],
                grids=[dict(node=names[node], shed_gwh=shed[i]) for i, node in enumerate(layout["grids"])],
                demands=[
                    dict(node=names[node], k=goods[k], unmet_unit_weeks=shortage[i])
                    for i, (node, k) in enumerate(layout["demands"])
                ],
                disposal=[
                    dict(node=names[node], k=goods[k], qty=disposed[i])
                    for i, (node, k) in enumerate(layout["stock_slots"])
                    if disposed[i] > 1e-6
                ],
            )
            rows.append(row)
            (out / f"episode-{episode}.json").write_text(json.dumps(dict(summary=row, weeks=weeks, layout=layout)))
            (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    print(f"Mean cost: {np.mean([r['cost_usd'] for r in rows]):,.0f} USD", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
