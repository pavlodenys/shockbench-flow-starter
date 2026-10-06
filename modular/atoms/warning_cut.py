"""Ship less through a strait whose early-warning score is high."""

import numpy as np

from modular.core import Atom, Param, register, seen  # build:strip


@register
class WarningCut(Atom):
    """A slot through a strait whose ``warning.score`` exceeds ``threshold`` ships ``1 - cut`` of its flow.

    A hypothesis, not a known gain: on 16 dev episodes of Small a score above 1.5 was followed by a closure within 4
    weeks 4 % of the time (base rate 1.8 %), so measure it (``python -m modular compare``). Reads ``warning.score``;
    an unobserved score counts as calm.
    """

    name = "warning_cut"
    ui = "Якщо раннє попередження про протоку вище порога, потік через неї зменшується. Гіпотеза: на даних Small ефекту не видно."  # noqa: E501
    stage = "scale"
    params = {
        "threshold": Param(1.5, -3.0, 5.0, "warning score above which the strait is treated as at risk"),
        "cut": Param(0.3, 0.0, 1.0, "share of the flow withheld when it is"),
    }

    def __init__(self, ctx, **values):
        super().__init__(ctx, **values)
        units = [tuple(u) for u in ctx.layout["warning_units"]]
        # strait position (row of graph_now.open) -> its unit in warning.score
        self.unit = np.array([units.index(("chokepoint", node)) for node in ctx.layout["chokepoints"]], dtype=int)

    def apply(self, plan, observation):
        score = seen(observation, "warning.score", -np.inf)[self.unit]
        factor = np.where(score > self.p["threshold"], 1.0 - self.p["cut"], 1.0)
        plan.flows = plan.flows * np.where(self.ctx.strait_matrix > 0, factor[None, :], 1.0).prod(axis=1)
        return plan
