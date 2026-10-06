"""A simple heuristic: ship what the route can carry this week, less through a strait that is partly closed.

Three rules, applied to every slot (a route and a commodity):
1. Start from the capacity the route has this week (``graph_now.u``), not its nominal one.
2. Ship nothing where ``action_mask`` is 0 (the route is prohibited).
3. A slot whose route passes a strait ships its capacity times that strait's open fraction.
Everything else stays at the maximum. It ignores stock, demand, warnings and announcements.
"""

import numpy as np


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        slots = static["action_slots"]
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        self.edge = np.asarray(slots["edge"], dtype=int)  # slot -> its edge
        self.nominal = np.array([u0[e] for e in self.edge], dtype=float)
        position = {node: i for i, node in enumerate(layout["chokepoints"])}  # strait node -> index in graph_now.open
        lane_straits = static["lanes"]["chokepoints"]  # lane -> the strait nodes it passes
        # slot -> the indices (in graph_now.open) of the straits its route passes; [] off any lane
        self.straits = [
            [position[node] for node in lane_straits[lane]] if lane is not None else [] for lane in slots["lane"]
        ]

    def act(self, observation):
        # 1. this week's capacity per slot; fall back to nominal where the value is not observed
        u = np.where(observation["graph_now.u.observed"] == 1, observation["graph_now.u"], np.inf)
        capacity = np.minimum(self.nominal, u[self.edge])
        # 2. nothing on prohibited routes
        flows = capacity * observation["action_mask"]
        # 3. less through a partly closed strait (1 open .. 0 closed; unobserved counts as open)
        is_open = np.where(observation["graph_now.open.observed"] == 1, observation["graph_now.open"], 1.0)
        for slot, straits in enumerate(self.straits):
            for strait in straits:
                flows[slot] *= np.clip(is_open[strait], 0.0, 1.0)
        return {"flows": flows}
