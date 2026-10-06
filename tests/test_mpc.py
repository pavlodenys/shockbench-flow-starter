"""MPC must respect chronology, inventory, separate terminals and shared resources."""

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.mine.agent import Agent as Mine
from agents.mpc.agent import Agent
from sbf_starter import env_id


sys.path.insert(0, str(Path(__file__).parents[1] / "examples/10_openevolve"))
from evaluator import read_params, training_savings


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
def test_mpc_runs_with_mask_and_preserves_nuclear(task):
    env = gym.make(env_id(task), entropy=202610141)
    try:
        obs, info = env.reset(seed=0, options={"episode": 0})
        config = agent_config_from_reset(env, obs, info)
        agent = Agent(config)
        for _ in range(4):
            action = agent.act(obs)
            flow = action["flows"]
            assert np.all(np.isfinite(flow)) and np.all(flow >= 0)
            assert np.all(flow <= agent.cap + 1e-7)
            assert np.all(flow[obs["action_mask"] == 0] == 0)
            np.testing.assert_array_equal(
                flow[agent.nuclear_slots], (agent.cap * obs["action_mask"])[agent.nuclear_slots]
            )
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        assert agent.mpc_solves > 0
        assert agent.mpc_failures == 0
    finally:
        env.close()


def test_zero_blend_is_v8_and_missing_stock_uses_base():
    env = gym.make(env_id("small"), entropy=202610141)
    try:
        obs, info = env.reset(seed=0)
        config = agent_config_from_reset(env, obs, info)
        agent, baseline = Agent(config), Mine(config)
        agent.options["mpc_blend"] = 0
        np.testing.assert_allclose(agent.act(obs)["flows"], baseline.act(obs)["flows"])
        agent.options["mpc_blend"] = 1
        obs["stock.qty.observed"][:] = 0
        np.testing.assert_allclose(agent.act(obs)["flows"], baseline.act(obs)["flows"])
    finally:
        env.close()


def test_mixed_fitness_cannot_hide_full_regression():
    baseline = {"tasks": {"small": {"cost_cents": [100]}, "full": {"cost_cents": [1000]}}}
    candidate = {"tasks": {"small": {"cost_cents": [70]}, "full": {"cost_cents": [1200]}}}
    score, networks = training_savings(candidate, baseline)
    assert score == pytest.approx(-20)
    assert networks["small"] == pytest.approx(30)


def test_mpc_parameters_reject_invalid_horizon():
    params = json.loads(Path("agents/mpc/params.json").read_text())
    assert read_params(f"PARAMS = {params!r}", "mpc") == params
    params["planning_horizon"] = 100
    with pytest.raises(ValueError):
        read_params(f"PARAMS = {params!r}", "mpc")


def test_cannot_dispatch_this_weeks_future_refill():
    env = gym.make(env_id("tiny"), entropy=202610141)
    try:
        obs, info = env.reset(seed=0)
        agent = Agent(agent_config_from_reset(env, obs, info))
        for _, _, source, _ in agent.routes:
            obs["stock.qty"][source] = 0
        agent.options["mpc_blend"] = 1
        action = agent.act(obs)["flows"]
        assert agent.mpc_solves == 1
        for slot, _, _, _ in agent.routes:
            assert action[slot] == 0
    finally:
        env.close()
