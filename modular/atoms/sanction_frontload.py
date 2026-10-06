"""Ship ahead of an announced sanction."""

import numpy as np

from modular.core import Atom, Param, action_mask, register  # build:strip


@register
class SanctionFrontload(Atom):
    """Raise a slot towards its capacity in the weeks before an announced prohibition takes effect.

    ``pending_prohibitions.*`` lists sanctions announced but not yet in force, each with its effective week. A slot
    whose route (any of its edges) carries such a prohibition for its commodity, taking effect within ``lead`` weeks,
    moves ``boost`` of the way from its current flow to this week's capacity: goods get across while the route is
    still allowed. Does nothing on a ``capacity`` base (already at the maximum): pair it with ``fraction`` below 1.
    """

    name = "sanction_frontload"
    ui = "За кілька тижнів до оголошеної санкції піднімає потік по маршрутах під забороною: вантаж встигає пройти. Корисно з fraction < 1."  # noqa: E501
    stage = "stock"
    params = {
        "lead": Param(3.0, 1.0, 12.0, "act this many weeks before the effective week"),
        "boost": Param(1.0, 0.0, 1.0, "share of the gap to capacity that is closed"),
    }

    def apply(self, plan, observation):
        ctx = self.ctx
        live = observation["pending_prohibitions.effective_week.observed"] == 1
        week = int(observation["week"][0])
        edge = observation["pending_prohibitions.edge"][live]
        comm = observation["pending_prohibitions.k"][live]
        due = observation["pending_prohibitions.effective_week"][live] - week
        soon = {(int(e), int(c)) for e, c, d in zip(edge, comm, due) if 0 <= d <= self.p["lead"]}
        if not soon:
            return plan
        hit = np.array([any((e, int(ctx.k[s])) in soon for e in ctx.route_edges[s]) for s in range(ctx.n_slots)])
        room = np.maximum(ctx.capacity * action_mask(observation, ctx.n_slots) - plan.flows, 0.0)
        plan.flows = plan.flows + self.p["boost"] * room * hit
        return plan
