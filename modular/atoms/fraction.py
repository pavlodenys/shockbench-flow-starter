"""Ship a fixed share of the base flow."""

from modular.core import Atom, Param, register  # build:strip


@register
class Fraction(Atom):
    """Every flow is multiplied by ``share``: the heuristic agent's ``fraction`` (one number for all slots).

    Below 1 it leaves headroom for atoms that raise flows later (``sanction_frontload``). Reads nothing.
    """

    name = "fraction"
    ui = "Відправляється лише задана частка базового потоку. Менше за 1 лишає запас для атомів, що піднімають потік."  # noqa: E501
    stage = "scale"
    params = {"share": Param(1.0, 0.0, 1.0, "share of the base flow that is shipped")}

    def apply(self, plan, observation):
        plan.flows = plan.flows * self.p["share"]
        return plan
