"""Safety properties of mine and its retained fuel fallback on public configs."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.mine import agent as policy
from sbf_starter import env_id


@pytest.fixture(params=["tiny", "small", "full"])
def scenario(request):
    env = gym.make(env_id(request.param), entropy=12345)
    obs, info = env.reset(seed=0, options={"episode": 0})
    config = agent_config_from_reset(env, obs, info)
    yield policy.Agent(config), obs, config
    env.close()


def test_shapes_and_route_prohibitions(scenario):
    agent, obs, config = scenario
    obs["action_mask"][::2] = 0
    flows = agent.act(obs)["flows"]
    assert flows.shape == tuple(config["spaces"]["action"]["flows"]["shape"])
    assert np.all(np.isfinite(flows)) and np.all(flows >= 0)
    assert np.all(flows[::2] == 0)
    assert np.all(flows <= agent.cap)


def test_final_goods_route_respects_observed_strait_open_fraction(scenario):
    agent, obs, _ = scenario
    # The fuel fallback keeps the openness rule; the joint plan can assign
    # another quantity using demand, stock and downstream capacity.
    choices = [(slot, positions) for slot, positions in enumerate(agent.final_route_chokepoints) if positions]
    if not choices:
        pytest.skip("this network has no final-goods route through a strait")
    slot, positions = choices[0]
    pos = positions[0]
    obs["action_mask"][slot] = 1
    obs["graph_now.open"][pos] = 0.1
    obs["graph_now.open.observed"][pos] = 1
    assert policy.FuelAgent.act(agent, obs)["flows"][slot] == pytest.approx(agent.cap[slot] * 0.01)
    obs["graph_now.open.observed"][pos] = 0
    assert policy.FuelAgent.act(agent, obs)["flows"][slot] == pytest.approx(agent.cap[slot])


def test_missing_inventory_keeps_other_template_requests(scenario):
    agent, obs, _ = scenario
    obs["stock.qty.observed"][:] = 0
    ordinary = np.array([not positions for positions in agent.final_route_chokepoints])
    np.testing.assert_array_equal(
        agent.act(obs)["flows"][ordinary], (agent.cap * obs["action_mask"])[ordinary]
    )


def test_solver_failure_keeps_external_requests(scenario, monkeypatch):
    agent, obs, _ = scenario
    monkeypatch.setattr(policy, "linprog", lambda *a, **kw: SimpleNamespace(success=False, status=2))
    flows = agent.act(obs)["flows"]
    external = [r[0] for r in agent.routes]
    np.testing.assert_array_equal(flows[external], (agent.cap * obs["action_mask"])[external])
    assert agent.failures == 1
    assert agent.chain_failures == 1


def test_full_grid_does_not_receive_more_than_consumption(scenario):
    agent, obs, _ = scenario
    for slot, group, dest in agent.internal:
        obs["stock.qty"][dest] = agent.grid_storage[dest]
    flows = policy.FuelAgent.act(agent, obs)["flows"]
    generation = agent._read(obs, "graph_now.grid.G_bar", agent.grid_nominal)
    for slot, group, dest in agent.internal:
        gi, _, _, share, _, _ = agent.groups[group]
        assert flows[slot] <= generation[gi] * share + 1e-7


def test_late_cargo_does_not_suppress_urgent_order(scenario):
    agent, obs, config = scenario
    # Isolate a direct LNG supply route, with an empty receiving grid/terminal.
    lng = config["static"]["commodities"]["id"].index("lng")
    slot, group, source, route = next(
        r for r in agent.routes if agent.slot_k[r[0]] == lng and len(r[3]) == 1
    )
    obs["action_mask"][:] = 0
    obs["action_mask"][slot] = 1
    obs["stock.qty"][agent.groups[group][-1]] = 0
    obs["stock.qty"][source] = agent.cap[slot]
    obs["pipeline.qty.observed"][:] = 0
    obs["queue_lots.qty.observed"][:] = 0
    obs["graph_now.u"][route[0]] = agent.cap[slot]
    obs["graph_now.tau"][route[0]] = 1
    obs["graph_now.u.observed"][route[0]] = 1
    obs["graph_now.tau.observed"][route[0]] = 1
    no_cargo = policy.FuelAgent.act(agent, obs)["flows"][slot]

    for key, value in (("edge", route[0]), ("k", lng), ("qty", 1e9), ("arrival_week", 18)):
        obs[f"pipeline.{key}"][0] = value
        obs[f"pipeline.{key}.observed"][0] = 1
    obs["pipeline.lane.observed"][0] = 0
    late_cargo = policy.FuelAgent.act(agent, obs)["flows"][slot]
    obs["pipeline.arrival_week"][0] = 1
    early_cargo = policy.FuelAgent.act(agent, obs)["flows"][slot]

    assert late_cargo == pytest.approx(no_cargo)
    assert late_cargo > early_cargo + 1


def test_nuclear_supply_is_preserved_with_large_initial_stock(scenario):
    agent, obs, config = scenario
    goods = config["static"]["commodities"]["id"]
    nuclear = np.array([goods[k] == "nucfuel" for k in agent.slot_k], dtype=bool)
    for _, _, k, _, _, indices in agent.groups:
        if goods[k] == "nucfuel":
            for index in indices:
                if index in agent.grid_storage:
                    obs["stock.qty"][index] = agent.grid_storage[index]
    # Preserve allowed nuclear requests even when the short-term LP sees ample
    # stock, but continue to respect route prohibitions. Tiny has no such fuel.
    nuclear_slots = np.flatnonzero(nuclear)
    if nuclear_slots.size:
        obs["action_mask"][nuclear_slots[0]] = 0
    flows = agent.act(obs)["flows"]
    np.testing.assert_array_equal(flows[nuclear], (agent.cap * obs["action_mask"])[nuclear])
