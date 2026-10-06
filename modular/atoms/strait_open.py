"""Scale a flow through a strait by the strait's open fraction to a power."""

import numpy as np

from modular.core import Atom, Param, register  # build:strip


@register
class StraitOpen(Atom):
    """A slot whose lane passes a strait ships (that strait's observed open fraction) ** power of its flow.

    Cargo sent into a closed strait queues there and pays holding cost; the rule is the heuristic agent's. With
    ``power`` 0 it does nothing; larger powers hold back more when a strait is only partly open. A strait whose
    ``graph_now.open`` value is not observed counts as open. Reads ``graph_now.open``.
    """

    name = "strait_open"
    ui = (
        "Потік через протоку множиться на (її відкритість)^степінь: чим закритіша протока, тим менше вантажу в неї йде."
    )
    stage = "scale"
    params = {"power": Param(1.0, 0.0, 5.0, "exponent applied to the open fraction")}

    def apply(self, plan, observation):
        is_open = np.where(
            observation["graph_now.open.observed"] == 1, np.maximum(observation["graph_now.open"], 0.0), 1.0
        )
        factor = is_open ** self.p["power"]  # per strait; 0 ** 0 is 1
        per_slot = np.where(self.ctx.strait_matrix > 0, factor[None, :], 1.0).prod(axis=1)
        plan.flows = plan.flows * per_slot
        return plan
