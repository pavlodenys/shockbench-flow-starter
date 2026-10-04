"""Announcements must change orders only when visible, relevant and timely."""

import copy

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.mine.agent import Agent as CurrentAgent
from agents.news_candidate.agent import Agent
from sbf_starter import env_id


@pytest.fixture(params=["tiny", "small", "full"])
def scenario(request):
    env = gym.make(env_id(request.param), entropy=314159)
    obs, info = env.reset(seed=0, options={"episode": 0})
    config = agent_config_from_reset(env, obs, info)
    for name in ("edge", "k", "effective_week"):
        obs[f"pending_prohibitions.{name}.observed"][:] = 0
    yield Agent(config), obs, config
    env.close()


def announce(obs, edge, k, week, index=0):
    for name, value in (("edge", edge), ("k", k), ("effective_week", week)):
        obs[f"pending_prohibitions.{name}"][index] = value
        obs[f"pending_prohibitions.{name}.observed"][index] = 1


def direct_route(agent, config):
    k = config["static"]["commodities"]["id"].index("lng")
    return next(r for r in agent.routes if agent.slot_k[r[0]] == k and len(r[3]) == 1)


def test_no_visible_news_matches_v3(scenario):
    agent, obs, config = scenario
    reference = CurrentAgent(config).act(copy.deepcopy(obs))["flows"]
    np.testing.assert_array_equal(agent.act(obs)["flows"], reference)


def test_relevant_news_increases_order(scenario):
    agent, obs, config = scenario
    slot, group, source, route = direct_route(agent, config)
    obs["action_mask"][:] = 0
    obs["action_mask"][slot] = 1
    obs["pipeline.qty.observed"][:] = 0
    obs["queue_lots.qty.observed"][:] = 0
    obs["graph_now.tau"][route[0]] = 1
    obs["graph_now.tau.observed"][route[0]] = 1
    obs["graph_now.u"][route[0]] = agent.cap[slot]
    obs["graph_now.u.observed"][route[0]] = 1
    obs["stock.qty"][source] = agent.cap[slot]
    gi, node, k, share, voll, indices = agent.groups[group]
    rate = share * agent._read(obs, "graph_now.grid.G_bar", agent.grid_nominal)[gi]
    obs["stock.qty"][indices] = 0
    lead = 2 + (agent.edges["head"][route[-1]] != node)
    obs["stock.qty"][indices[0]] = rate * (lead + agent.reserve + 1)
    before = agent.act(obs)["flows"]
    announce(obs, route[0], k, int(obs["week"][0]) + 3)
    after = agent.act(obs)["flows"]
    assert after[slot] > before[slot]
    assert np.isfinite(after).all() and (after >= 0).all() and (after <= agent.cap).all()
    assert (after[obs["action_mask"] == 0] == 0).all()


def test_duplicates_missing_fields_and_expired_news(scenario):
    agent, obs, config = scenario
    slot, group, source, route = direct_route(agent, config)
    obs["action_mask"][:] = 0
    obs["action_mask"][slot] = 1
    week = int(obs["week"][0])
    announce(obs, route[0], int(agent.slot_k[slot]), week + 3)
    args = (week, agent.nominal_tau, {}, agent.nominal_cap)
    reserve = agent._news_reserve(obs, *args)
    assert reserve[group] == 3
    announce(obs, route[0], int(agent.slot_k[slot]), week + 3, index=1)
    np.testing.assert_array_equal(agent._news_reserve(obs, *args), reserve)
    obs["pending_prohibitions.k.observed"][:] = 0
    assert not agent._news_reserve(obs, *args).any()
    obs["pending_prohibitions.k.observed"][:2] = 1
    obs["pending_prohibitions.effective_week"][:2] = week
    assert not agent._news_reserve(obs, *args).any()
    del obs["pending_prohibitions.edge"]
    assert not agent._news_reserve(obs, *args).any()


def test_late_crossing_does_not_trigger_stockpiling(scenario):
    agent, obs, config = scenario
    slot, group, source, route = next(r for r in agent.routes if len(r[3]) > 1)
    obs["action_mask"][:] = 0
    obs["action_mask"][slot] = 1
    week = int(obs["week"][0])
    tau = agent.nominal_tau.copy()
    tau[route[0]] = 4
    announce(obs, route[1], int(agent.slot_k[slot]), week + 4)
    assert not agent._news_reserve(obs, week, tau, {}, agent.nominal_cap).any()


def test_nuclear_orders_stay_at_baseline(scenario):
    agent, obs, config = scenario
    week = int(obs["week"][0])
    for i, (slot, group, source, route) in enumerate(agent.routes):
        announce(obs, route[0], int(agent.slot_k[slot]), week + 3, index=i)
    flows = agent.act(obs)["flows"]
    nuclear = agent.nuclear_slots
    np.testing.assert_array_equal(flows[nuclear], (agent.cap * obs["action_mask"])[nuclear])
