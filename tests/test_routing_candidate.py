"""Route changes must preserve requests and respect deadline/visibility limits."""

import gymnasium as gym
import numpy as np
import pytest
from shockbench_flow_gym import agent_config_from_reset

from agents.routing_candidate.agent import Agent
from sbf_starter import env_id


@pytest.fixture
def fork():
    agent = Agent.__new__(Agent)
    agent.T = 30
    agent.cap = np.array([100.0, 100.0])
    agent.slot_k = np.array([0, 0])
    agent.edges = {"tail": [0, 1, 2, 1, 3], "head": [1, 2, 4, 3, 4]}
    agent.chk_pos = {1: 0, 2: 1, 3: 2}
    agent.groups = [(0, 4)]
    agent.value = np.array([1.0])
    agent.route_choices = {(0, 0): [(0, [0, 1, 2]), (1, [0, 3, 4])]}
    agent.slot_edge = np.array([0, 0])
    agent.slot_paths = [[0, 1, 2], [0, 3, 4]]
    agent.slot_sources = [0, 0]
    agent.lanes = {"edges": agent.slot_paths}
    agent.pool = ["tb"]
    agent.lot_keys = [(3, 0, 1, 4)]
    agent.reroute_weeks = 0
    agent.reroute_requested_qty = 0.0
    obs = {
        "action_mask": np.ones(2),
        "graph_now.open": np.ones(3),
        "graph_now.open.observed": np.ones(3),
        "graph_now.kappa.tb": np.full(3, 100.0),
        "graph_now.kappa.tb.observed": np.ones(3),
        "stock.qty": np.array([10000.0]),
        "stock.qty.observed": np.ones(1),
        "queue_lots.qty": np.zeros((1, 1)),
        "queue_lots.qty.observed": np.ones((1, 1)),
    }
    for name in ("edge", "k", "lane", "qty"):
        obs[f"pipeline.{name}"] = np.zeros(1)
        obs[f"pipeline.{name}.observed"] = np.ones(1)
    for name in ("edge", "k", "effective_week"):
        obs[f"pending_prohibitions.{name}"] = np.zeros(4, dtype=int)
        obs[f"pending_prohibitions.{name}.observed"] = np.zeros(4, dtype=int)
    return agent, obs


def run(fork, flows=(80.0, 20.0), waits=None, price=None, capacity=None):
    agent, obs = fork
    return agent._reroute(
        np.array(flows),
        obs,
        1,
        np.array([1, 1, 1, 2, 2]),
        waits or {},
        np.full(5, 100.0) if capacity is None else capacity,
        np.ones(5) if price is None else price,
        np.zeros((5, 1)),
    )


def announce(obs, edge, effective):
    for name, value in (("edge", edge), ("k", 0), ("effective_week", effective)):
        obs[f"pending_prohibitions.{name}"][0] = value
        obs[f"pending_prohibitions.{name}.observed"][0] = 1


def test_no_congestion_or_deadline_keeps_original(fork):
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_closed_branch_shifts_half_and_preserves_total(fork):
    agent, obs = fork
    obs["graph_now.open"][1] = 0
    result = run(fork)
    np.testing.assert_array_equal(result, [40, 60])
    assert sum(result) == 100
    assert agent.reroute_weeks == 1 and agent.reroute_requested_qty == 40


def test_unobserved_closure_is_ignored(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    obs["graph_now.open.observed"][1] = 0
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_deadline_at_entry_week_counts_as_too_late(fork):
    _, obs = fork
    announce(obs, 1, 2)
    np.testing.assert_array_equal(run(fork), [40, 60])
    obs["pending_prohibitions.effective_week"][0] = 3
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_departure_before_first_edge_ban_is_preserved(fork):
    _, obs = fork
    announce(obs, 0, 2)
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_hidden_or_withdrawn_notice_does_not_persist(fork):
    _, obs = fork
    announce(obs, 1, 2)
    np.testing.assert_array_equal(run(fork), [40, 60])
    obs["pending_prohibitions.k.observed"][:] = 0
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_queue_delay_can_trigger_a_detour(fork):
    np.testing.assert_array_equal(run(fork, waits={(2, 0): 4}), [40, 60])


def test_destination_capacity_limits_transfer(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    np.testing.assert_array_equal(run(fork, flows=(80.0, 90.0)), [70, 100])


def test_forbidden_or_expensive_alternative_is_not_used(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    obs["action_mask"][1] = 0
    np.testing.assert_array_equal(run(fork, flows=(80.0, 0.0)), [80, 0])
    obs["action_mask"][1] = 1
    np.testing.assert_array_equal(run(fork, price=np.array([1, 1, 1, 10, 10])), [80, 20])


def test_no_safe_alternative_preserves_original(fork):
    _, obs = fork
    obs["graph_now.open"][1:] = 0
    np.testing.assert_array_equal(run(fork), [80, 20])


def test_positive_but_narrow_exit_limits_extra_dispatch(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    # Arrival at edge 3 is next week: 2 * 15 capacity, minus 20 already requested.
    np.testing.assert_array_equal(run(fork, capacity=np.array([100, 100, 100, 15, 100])), [70, 30])


def test_queued_cargo_blocks_a_nominally_faster_alternative(fork):
    _, obs = fork
    obs["queue_lots.qty"][:] = 1000
    result = run(fork, waits={(2, 0): 4})
    assert result[1] <= 20 and sum(result) == 100
    obs["queue_lots.qty.observed"][:] = 0
    np.testing.assert_array_equal(run(fork, waits={(2, 0): 4}), [40, 60])


def test_inflight_cargo_reserves_downstream_capacity(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    obs["pipeline.lane"][:] = 1
    obs["pipeline.qty"][:] = 190
    # Edge 3 has 200 service units before arrival; 190 inbound + 20 new fills it.
    np.testing.assert_array_equal(run(fork), [80, 20])
    obs["pipeline.qty.observed"][:] = 0
    np.testing.assert_array_equal(run(fork), [40, 60])


def test_shared_pool_capacity_limits_extra_dispatch(fork):
    _, obs = fork
    obs["graph_now.open"][1] = 0
    obs["graph_now.kappa.tb"][2] = 6
    # Alternative reaches node 3 after 3 weeks: 24 service units, 20 reserved.
    np.testing.assert_array_equal(run(fork), [76, 24])


def test_competing_group_cannot_reserve_same_exit_twice(fork):
    agent, obs = fork
    agent.cap = np.full(4, 100.0)
    agent.slot_k = np.zeros(4, dtype=int)
    agent.slot_edge = np.zeros(4, dtype=int)
    agent.slot_paths *= 2
    agent.slot_sources *= 2
    agent.groups *= 2
    agent.route_choices[0, 1] = [(2, [0, 1, 2]), (3, [0, 3, 4])]
    obs["action_mask"] = np.ones(4)
    obs["graph_now.open"][1] = 0
    result = run(fork, flows=(40, 10, 40, 10), capacity=np.array([100, 100, 100, 15, 100]))
    assert result[1] + result[3] == 30
    assert sum(result[:2]) == sum(result[2:]) == 50


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
def test_real_config_shapes_masks_and_nuclear_supply(task):
    env = gym.make(env_id(task), entropy=202610040)
    try:
        obs, info = env.reset(seed=0, options={"episode": 0})
        config = agent_config_from_reset(env, obs, info)
        agent = Agent(config)
        reserved = []
        routing_state = agent._routing_state

        def capture_reservations(flows, observation, capacity):
            reserved.append(flows.copy())
            return routing_state(flows, observation, capacity)

        agent._routing_state = capture_reservations
        obs["action_mask"][::3] = 0
        flows = agent.act(obs)["flows"]
        np.testing.assert_array_equal(reserved[0][agent.nuclear_slots], flows[agent.nuclear_slots])
        assert flows.shape == env.action_space["flows"].shape
        assert np.isfinite(flows).all() and (flows >= 0).all() and (flows <= agent.cap).all()
        assert (flows[obs["action_mask"] == 0] == 0).all()
        np.testing.assert_array_equal(flows[agent.nuclear_slots], (agent.cap * obs["action_mask"])[agent.nuclear_slots])
        # Every choice group shares the exact first edge, commodity and fuel sink.
        for (edge, group), choices in agent.route_choices.items():
            assert all(route[0] == edge for _, route in choices)
            assert len({agent.slot_k[slot] for slot, _ in choices}) == 1
    finally:
        env.close()
