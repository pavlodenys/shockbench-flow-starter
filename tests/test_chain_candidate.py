"""Feasibility and observation boundaries of the experimental supply-chain MPC."""

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.chain_candidate import agent as candidate
from agents.mine import agent as mine
from sbf_starter import env_id


@pytest.fixture(params=[candidate, mine], ids=["candidate", "mine"])
def policy(request):
    return request.param


@pytest.fixture(params=["tiny", "small", "full"])
def scenario(request, policy):
    env = gym.make(env_id(request.param), entropy=202610071)
    obs, info = env.reset(seed=0, options={"episode": 0})
    config = agent_config_from_reset(env, obs, info)
    yield policy.Agent(config), obs, config, policy
    env.close()


def test_first_dispatch_respects_stock_and_shared_edge_capacity(scenario):
    agent, obs, _, policy = scenario
    baseline = policy.FuelAgent.act(agent, obs)["flows"]
    plan = agent._chain_plan(obs, baseline)
    assert plan is not None, agent.last_chain_diagnostics
    sources = {}
    entries = {}
    for slot, qty in enumerate(plan):
        edge, k = int(agent.slot_edge[slot]), int(agent.slot_k[slot])
        pair = (agent.edges["tail"][edge], k)
        sources[pair] = sources.get(pair, 0.0) + qty
        entries[edge] = entries.get(edge, 0.0) + qty
    for pair, total in sources.items():
        assert total <= obs["stock.qty"][agent.stock_index[pair]] + 1e-5
    for edge, total in entries.items():
        assert total <= obs["graph_now.u"][edge] + 1e-5


def test_future_arrivals_and_production_cannot_be_dispatched_today(scenario):
    agent, obs, _, policy = scenario
    obs["stock.qty"][:] = 0
    # Pipeline and WIP remain visible. They may serve demand on arrival but
    # cannot supply any current dispatch, which precedes arrivals/production.
    baseline = policy.FuelAgent.act(agent, obs)["flows"]
    plan = agent._chain_plan(obs, baseline)
    assert plan is not None, agent.last_chain_diagnostics
    np.testing.assert_allclose(plan, 0.0, atol=1e-6)


def test_prohibitions_and_nuclear_requests_survive_joint_planning(scenario):
    agent, obs, config, _ = scenario
    obs["action_mask"][::2] = 0
    action = agent.act(obs)
    flows = action["flows"]
    assert flows.shape == tuple(config["spaces"]["action"]["flows"]["shape"])
    assert np.all(np.isfinite(flows)) and np.all(flows >= 0)
    assert np.all(flows[::2] == 0)
    assert np.all(flows <= agent.cap + 1e-6)
    np.testing.assert_allclose(
        flows[agent.nuclear_slots], (agent.cap * obs["action_mask"])[agent.nuclear_slots]
    )


def test_hidden_forecast_values_do_not_enter_plan(scenario):
    agent, obs, _, _ = scenario
    obs["demand_forecast.qty.observed"][:] = 0
    obs["demand_forecast.qty"][:] = np.inf
    flows = agent.act(obs)["flows"]
    assert np.all(np.isfinite(flows))
    assert agent.chain_failures == 0, agent.last_chain_diagnostics


def test_missing_inventory_uses_established_fallback(scenario):
    agent, obs, _, policy = scenario
    obs["stock.qty.observed"][:] = 0
    expected = policy.FuelAgent.act(agent, obs)["flows"]
    np.testing.assert_allclose(agent.act(obs)["flows"], expected)
    assert agent.chain_failures == 1
