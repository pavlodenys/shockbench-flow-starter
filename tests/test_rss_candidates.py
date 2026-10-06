"""Transport timing and policy safety on the public network configurations."""

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.eta_candidate.agent import Agent as EtaAgent
from agents.lng_candidate.agent import Agent as LngAgent
from sbf_starter import env_id


@pytest.fixture(params=["tiny", "small", "full"])
def scenario(request):
    env = gym.make(env_id(request.param), entropy=202610047)
    obs, info = env.reset(seed=0, options={"episode": 0})
    yield obs, agent_config_from_reset(env, obs, info)
    env.close()


@pytest.mark.parametrize("cls", [LngAgent, EtaAgent])
def test_candidate_masks_nuclear_and_finite_actions(scenario, cls):
    obs, config = scenario
    agent = cls(config)
    obs["action_mask"][::3] = 0
    flows = agent.act(obs)["flows"]
    assert flows.shape == agent.cap.shape
    assert np.all(np.isfinite(flows))
    assert np.all((flows >= 0) & (flows <= agent.cap + 1e-8))
    assert np.all(flows[::3] == 0)
    np.testing.assert_array_equal(flows[agent.nuclear_slots], (agent.cap * obs["action_mask"])[agent.nuclear_slots])


def test_terminal_stock_requires_dispatch_and_transit(scenario):
    obs, config = scenario
    agent = LngAgent(config)
    g = next((g for g in agent.ration_floor if any(row[1] == g for row in agent.internal)), None)
    if g is None:
        pytest.skip("This network has no terminal-to-grid transfer for a rationed fuel")
    _, grid, k, _, _, indices = agent.groups[g]
    stock = np.zeros_like(obs["stock.qty"])
    for i in indices:
        if i != agent.stock_index[grid, k]:
            stock[i] = 1000.0
    caps = np.full(len(agent.edges["id"]), 100.0)
    tau = np.ones_like(caps)
    mask = np.ones_like(obs["action_mask"])
    result = agent._grid_supply(g, stock, [], caps, tau, mask, 4)
    assert result[1] == 0.0
    assert 0.0 < result[2] <= 100.0
    assert result[3] > result[2]
    mask[:] = 0
    np.testing.assert_array_equal(agent._grid_supply(g, stock, [], caps, tau, mask, 4), 0.0)


def test_forecast_splits_exit_queue_and_respects_pipeline_time(scenario):
    obs, config = scenario
    agent = EtaAgent(config)
    week = int(obs["week"][0])
    obs["queue_lots.qty"][:] = 0
    obs["queue_lots.qty.observed"][:] = 0
    obs["pipeline.qty.observed"][:] = 0
    obs["action_mask"][:] = 0
    obs["graph_now.prohibited"][:] = 0
    obs["graph_now.prohibited.observed"][:] = 1
    obs["closure_end.end_week.observed"][:] = 0
    obs["pending_prohibitions.effective_week.observed"][:] = 0
    for pool in set(agent.pool):
        obs["graph_now.kappa." + pool][:] = 1e9
        obs["graph_now.kappa." + pool + ".observed"][:] = 1
    # Isolate a final lane edge with no detour charge.
    slot, g, _, path = next(
        r
        for r in agent.routes
        if len(r[3]) > 1 and agent.edges["tail"][r[3][-1]] in agent.chk_pos and not agent.fleet_terms.get(r[3][-1])
    )
    edge, k = path[-1], int(agent.slot_k[slot])
    lane = agent.slot_lane[slot]
    node = agent.edges["tail"][edge]
    key = (node, k, lane, edge)
    if agent.lot_keys is not None:
        index = next(i for i, row in enumerate(agent.lot_keys) if tuple(row) == key)
        obs["queue_lots.qty"][index, 0] = 25.0
        obs["queue_lots.qty.observed"][index, 0] = 1
    else:
        for name, value in zip(("chokepoint", "k", "lane", "next_edge", "arrival_week", "qty"), (*key, 1, 25.0)):
            obs["queue_lots." + name][0] = value
            obs["queue_lots." + name + ".observed"][0] = 1
    caps = np.full(len(agent.edges["id"]), 1e9)
    caps[edge] = 10.0
    tau = np.ones_like(caps)
    bounds = [(0.0, 0.0)] * len(agent.routes)
    arrivals, _, _ = agent._forecast(obs, week, caps, tau, bounds)
    expected_first = 2 + (agent.edges["head"][edge] != agent.groups[g][1])
    assert arrivals[g] == [(expected_first, 10.0), (expected_first + 1, 10.0), (expected_first + 2, 5.0)]
    obs["queue_lots.qty.observed"][:] = 0
    previous = path[-2]
    for name, value in (("edge", previous), ("k", k), ("lane", lane), ("qty", 25.0), ("arrival_week", week + 3)):
        obs["pipeline." + name][0] = value
        obs["pipeline." + name + ".observed"][0] = 1
    arrivals, _, _ = agent._forecast(obs, week, caps, tau, bounds)
    assert arrivals[g] == [(expected_first + 3, 10.0), (expected_first + 4, 10.0), (expected_first + 5, 5.0)]
    obs["pipeline.arrival_week.observed"][0] = 0
    arrivals, _, _ = agent._forecast(obs, week, caps, tau, bounds)
    assert arrivals[g] == []
