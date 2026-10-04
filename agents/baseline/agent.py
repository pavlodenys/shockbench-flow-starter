"""Your agent. The scorer builds ``Agent(config)`` once per episode and calls ``act(observation)`` every week.

As shipped it sends the maximum: every route allowed this week ships its nominal capacity. On the server only the
standard library, numpy, scipy and torch (CPU) exist. Load your files at module level, relative to ``HERE``: the CPU
budget does not meter the import. A week whose ``act`` raises or returns a malformed action is played by the naive
rule. The rules: docs/GUIDE.md. Check it with ``uv run sbf check <name>``.
"""

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


class Agent:
    def __init__(self, config=None):
        static = config["static"]  # the network's tables: nodes, edges, lanes, commodities, action slots, ...
        action = config["spaces"]["action"]  # the shape of each part of the action
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        self.capacity = np.array([u0[e] for e in static["action_slots"]["edge"]], dtype=float)
        self.override_qty = np.zeros(action["override_qty"]["shape"])
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)  # 0: the default release
        self.rng = np.random.default_rng(config["policy_seed"])  # unused here; seed all randomness from policy_seed

    def act(self, observation):
        # observation: numpy arrays by name ("stock.qty", "graph_now.u", ...); action_mask is 0 on sanctioned routes
        flows = self.capacity * observation["action_mask"]
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
