"""Neutral weights preserve baseline; active extensions respect physical budgets."""

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.adaptive_reserve import agent as policy
from sbf_starter import env_id


def test_all_extension_weights_start_at_zero():
    assert all(value == 0 for key, value in policy.PARAMS.items() if key.endswith("_weight"))


def test_adaptive_lp_share_is_bounded_and_lng_specific():
    agent = policy.Agent.__new__(policy.Agent)
    agent.blend = 0.7
    agent.options = {key: 0.0 for key in policy.EXTENSION_BOUNDS}
    agent.lng_groups = {0}
    agent._reserve_signals = lambda *args: np.array([3.0, 4.0, 3.0])
    args = (0, 30, 10.0, 20.0, [], [5.0], [(0.0, 10.0)])
    assert agent._adaptive_blend(*args) == 0.7
    agent.options["lp_shortage_weight"] = 1.0
    assert agent._adaptive_blend(*args) == 1.0
    agent.options["lp_shortage_weight"] = 0.0
    agent.options["lp_delay_weight"] = -1.0
    assert agent._adaptive_blend(*args) == 0.0
    agent.options["lp_delay_weight"] = 0.0
    agent.options["lng_lp_weight"] = -0.2
    assert agent._adaptive_blend(*args) == pytest.approx(0.5)
    assert agent._adaptive_blend(1, *args[1:]) == 0.7


def test_lng_reserve_offset_does_not_change_other_fuels():
    agent = policy.Agent.__new__(policy.Agent)
    agent.reserve = 4.0
    agent.reserve_min, agent.reserve_max = 0.0, 8.0
    agent.reserve_weights = np.zeros(3)
    agent.options = {key: 0.0 for key in policy.EXTENSION_BOUNDS}
    agent.options["lng_reserve_weight"] = 2.0
    agent.lng_groups = {0}
    agent._reserve_signals = lambda *args: np.zeros(3)
    args = (30, 10, 20, [], [5], [(0, 10)])
    assert agent._adaptive_reserve(0, *args) == 6
    assert agent._adaptive_reserve(1, *args) == 4


def test_terminal_arrival_cannot_supply_grid_in_same_week():
    agent = policy.Agent.__new__(policy.Agent)
    agent.groups = [(0, 2, 0, 1.0, 1.0, [0, 1])]
    agent.stock_index = {(1, 0): 0, (2, 0): 1}
    agent.edges = {"tail": [0, 1]}
    agent.slot_edge = np.array([0, 1])
    agent.cap = np.array([100.0, 100.0])
    agent.internal = [(1, 0, 1)]
    agent.routes = [(0, 0, 2, [0])]
    agent.lng_groups = {0}
    agent._lng_events = {0: []}
    agent._lng_dispatch_events = {0: [(1, 1, 100.0)]}
    supply, fractions = agent._lng_curves(np.zeros(3), np.ones(2) * 100, np.ones(2), np.ones(2), 4, [(0, 100)])
    np.testing.assert_array_equal(supply[0], 0)
    np.testing.assert_array_equal(fractions[0], [0, 0, 0, 1, 1])


def test_lng_terminal_stock_does_not_hide_grid_coverage_deficit():
    agent = policy.Agent.__new__(policy.Agent)
    agent.lng_groups, agent.lng_floor = {0}, {0: 5.0}
    agent.options = {key: 0.0 for key in policy.EXTENSION_BOUNDS}
    assert agent._coverage_target(0, 4, 10, 1000, [], {}, 4) == 0
    agent.options.update(lng_terminal_weight=1, lng_ration_weight=1)
    accessible = {0: np.array([0, 0, 0, 10, 10])}
    assert agent._coverage_target(0, 4, 10, 1000, [], accessible, 4) == 35
    agent.options["lng_terminal_weight"] = 0.5
    assert agent._coverage_target(0, 4, 10, 1000, [], accessible, 4) == 17.5


def semi_planner():
    agent = policy.Agent.__new__(policy.Agent)
    agent.T, agent.reserve = 10, 4.0
    agent.options = {key: 0.0 for key in policy.EXTENSION_BOUNDS}
    agent.options.update(semi_allocation_weight=1.0, semi_shortage_weight=2.0)
    agent.cap = np.array([100.0, 100.0])
    agent.value = np.array([10000.0])
    agent.slot_edge = np.array([0, 1])
    agent.commodity_names = ["chip_le"]
    agent.sink_position = {(1, 0): 0, (2, 0): 1}
    agent.sink_penalty = np.array([50000.0, 50000.0])
    agent.semi_routes = [(0, 0, 1, 1, 0, [0]), (1, 0, 2, 2, 0, [1])]
    obs = {
        "week": np.array([1]),
        "stock.qty": np.array([100.0, 0.0, 1000.0]),
        "stock.qty.observed": np.ones(3),
        "demand_forecast.qty": np.ones((2, 8)) * 100,
        "demand_forecast.qty.observed": np.ones((2, 8)),
        "backlog.qty": np.zeros(2),
        "backlog.qty.observed": np.ones(2),
    }
    return agent, obs


def test_semiconductor_priorities_favor_starved_market_without_creating_stock():
    agent, obs = semi_planner()
    flows = agent._semiconductor_priorities(np.ones(2) * 100, obs, np.ones(2), np.ones(2) * 100)
    assert flows[0] > flows[1]
    assert flows.sum() == pytest.approx(100)
    assert np.all((flows >= 0) & (flows <= 100))
    # Zero request (including a forbidden route) cannot receive a priority allocation.
    flows = agent._semiconductor_priorities(np.array([0.0, 100.0]), obs, np.ones(2), np.ones(2) * 100)
    assert flows[0] == 0


def test_unknown_forecast_preserves_its_request_and_stock_reservation():
    agent, obs = semi_planner()
    obs["demand_forecast.qty.observed"][0] = 0
    flows = agent._semiconductor_priorities(np.ones(2) * 100, obs, np.ones(2), np.ones(2) * 100)
    assert flows[0] == 100
    assert flows[1] == 0


def test_weighted_allocation_caps_requests_and_uses_remaining_budget():
    values = policy.Agent._weighted_allocation([1, 100], [4, -4], 50)
    np.testing.assert_allclose(values, [1, 49])
    np.testing.assert_array_equal(policy.Agent._weighted_allocation([0, 0], [0, 0], 100), 0)


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
def test_all_extensions_active_keep_valid_actions_and_nuclear_supply(task):
    env = gym.make(env_id(task), entropy=202610091)
    try:
        obs, info = env.reset(seed=0, options={"episode": 0})
        agent = policy.Agent(agent_config_from_reset(env, obs, info))
        agent.options.update(
            lp_shortage_weight=0.5,
            lp_delay_weight=-0.25,
            lng_reserve_weight=1,
            lng_ration_weight=1,
            lng_terminal_weight=1,
            lng_lp_weight=0.2,
            semi_allocation_weight=1,
            semi_shortage_weight=1,
            semi_value_weight=1,
            semi_delay_weight=1,
        )
        obs["action_mask"][::3] = 0
        flows = agent.act(obs)["flows"]
        assert np.all(np.isfinite(flows))
        assert np.all((flows >= 0) & (flows <= agent.cap + 1e-7))
        assert np.all(flows[::3] == 0)
        np.testing.assert_array_equal(flows[agent.nuclear_slots], (agent.cap * obs["action_mask"])[agent.nuclear_slots])
        assert agent.failures == 0
    finally:
        env.close()
