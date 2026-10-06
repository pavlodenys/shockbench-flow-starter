"""Adaptive reserve must preserve v8 at zero weights and respect route eligibility."""

import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.adaptive_reserve import agent as adaptive
from agents.mine import agent as mine
from sbf_starter import env_id


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
def test_zero_weights_preserve_v8_actions(task):
    env = gym.make(env_id(task), entropy=202610071)
    try:
        obs, info = env.reset(seed=0, options={"episode": 0})
        config = agent_config_from_reset(env, obs, info)
        new, baseline = adaptive.Agent(config), mine.Agent(config)
        for _ in range(3):
            np.testing.assert_allclose(new.act(obs)["flows"], baseline.act(obs)["flows"], atol=1e-7)
            obs, _, terminated, truncated, _ = env.step(baseline.act(obs))
            if terminated or truncated:
                break
        obs["action_mask"][::2] = 0
        for weights in ([2, 2, 2], [-2, -2, -2]):
            new.reserve_weights = np.array(weights, dtype=float)
            flows = new.act(obs)["flows"]
            assert np.all(np.isfinite(flows)) and np.all(flows >= 0)
            assert np.all(flows <= new.cap + 1e-7)
            assert np.all(flows[::2] == 0)
            np.testing.assert_array_equal(flows[new.nuclear_slots], (new.cap * obs["action_mask"])[new.nuclear_slots])
        obs["stock.qty.observed"][:] = 0
        np.testing.assert_array_equal(new.act(obs)["flows"], new.cap * obs["action_mask"])
    finally:
        env.close()


def planner(weights):
    agent = adaptive.Agent.__new__(adaptive.Agent)
    agent.reserve = 4.0
    agent.reserve_min, agent.reserve_max = 1.0, 10.0
    agent.reserve_weights = np.array(weights, dtype=float)
    agent.options = {key: 0.0 for key in adaptive.EXTENSION_BOUNDS}
    agent.lng_groups = set()
    agent.nominal_tau = np.array([2.0, 3.0, 1.0])
    agent.routes = [(0, 0, 0, [0]), (1, 0, 0, [1]), (2, 1, 0, [2])]
    return agent


def reserve(agent, eta=(5, 9, 1), bounds=((0, 10), (0, 10), (0, 10)), arrivals=(), remaining=30):
    return agent._adaptive_reserve(0, remaining, 10.0, 20.0, arrivals, eta, bounds)


@pytest.mark.parametrize("weights,expected", [([1, 0, 0], 7), ([0, 1, 0], 8), ([0, 0, 1], 7)])
def test_signals_increase_reserve_in_weeks(weights, expected):
    assert reserve(planner(weights)) == pytest.approx(expected)


def test_only_timely_inbound_stock_reduces_shortage_signal():
    agent = planner([0, 0, 1])
    assert reserve(agent, arrivals=[(5, 30)]) == 4
    assert reserve(agent, arrivals=[(6, 30)]) == 7


def test_closed_unreachable_and_other_group_routes_are_excluded():
    agent = planner([1, 1, 0])
    assert reserve(agent, bounds=((0, 10), (0, 0), (0, 10))) == 7
    assert reserve(agent, eta=(5, 999, 999)) == 7
    assert reserve(agent, bounds=((0, 0), (0, 0), (0, 10))) == 4


def test_negative_weights_bounds_and_end_of_episode():
    assert reserve(planner([-2, -2, -2])) == 1
    assert reserve(planner([2, 2, 2])) == 10
    assert reserve(planner([2, 2, 2]), remaining=2) == 2


def test_adaptive_parser_and_frozen_candidate_source(tmp_path):
    path = Path(__file__).parents[1] / "examples/10_openevolve/evaluator.py"
    spec = importlib.util.spec_from_file_location("adaptive_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    params = dict(adaptive.PARAMS)
    assert evaluator.read_params(f"PARAMS = {params!r}", "adaptive") == params
    params["reserve_min"], params["reserve_max"] = 8, 3
    with pytest.raises(ValueError, match="reserve_min"):
        evaluator.read_params(f"PARAMS = {params!r}", "adaptive")
    params = dict(adaptive.PARAMS)
    params["delay_weight"] = float("inf")
    with pytest.raises(ValueError):
        evaluator.read_params(f"PARAMS = {params!r}", "adaptive")
    (tmp_path / "candidate-source").mkdir()
    source = tmp_path / "candidate-source/agent.py"
    source.write_text("adaptive source")
    folder = evaluator.write_candidate(tmp_path, adaptive.PARAMS)
    assert (folder / "agent.py").read_bytes() == source.read_bytes()
    assert json.loads((folder / "params.json").read_text()) == adaptive.PARAMS
