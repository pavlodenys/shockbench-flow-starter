"""The empty agent: reads nothing and ships nothing.

Every week it returns zero flows, so no route is used. It is a floor for comparison and a blank page to start
from: ``config`` (once per episode) and ``observation`` (every week) are unused here. The rules: docs/GUIDE.md.
Check it with ``uv run sbf check empty``.
"""

import numpy as np


class Agent:
    def __init__(self, config=None):
        self.slots = len(config["static"]["action_slots"]["edge"])  # one flow per (edge, commodity, lane) slot

    def act(self, observation):
        return {"flows": np.zeros(self.slots)}
