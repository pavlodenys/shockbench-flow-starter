"""Do not ship to a market more than its forecast demand needs."""

import numpy as np

from modular.core import Atom, Param, register, seen  # build:strip


@register
class DemandCap(Atom):
    """Cap the total sent towards each demand node and commodity at what the forecast still needs there.

    Need = ``cover`` weeks of the forecast (mean of the next ``cover`` forecast weeks, at most 8) plus the backlog,
    minus the stock on hand and the shipments on their last leg; the cap is ``slack`` times that, shared in
    proportion among the slots that reach the market. Saves holding cost; too small a ``cover`` causes shortages.
    Reads ``demand_forecast.qty``, ``backlog.qty``, ``stock.qty`` and ``pipeline.*``. Cargo still queued at a strait
    or on an earlier leg is not counted, so the cap errs on the generous side.
    """

    name = "demand_cap"
    ui = "Загальна відправка до ринку не перевищує того, що ще потрібно за прогнозом попиту з урахуванням запасу, боргу й вантажу в дорозі."  # noqa: E501
    stage = "stock"
    params = {
        "cover": Param(4.0, 1.0, 8.0, "weeks of forecast demand to keep covered"),
        "slack": Param(1.5, 1.0, 5.0, "multiplier on the computed need"),
    }

    def __init__(self, ctx, **values):
        super().__init__(ctx, **values)
        layout, static = ctx.layout, ctx.static
        self.demands = [tuple(d) for d in layout["demands"]]  # rows of backlog.qty and demand_forecast.qty: (node, k)
        stock_row = {tuple(sk): i for i, sk in enumerate(layout["stock_slots"])}
        self.stock_row = np.array([stock_row[d] for d in self.demands], dtype=int)
        self.members = [np.flatnonzero((ctx.dest == n) & (ctx.k == k)) for n, k in self.demands]
        self.edge_head = np.asarray(static["edges"]["head"], dtype=int)

    def apply(self, plan, observation):
        forecast = seen(observation, "demand_forecast.qty")  # (demands, 8)
        weeks = max(1, min(forecast.shape[1], int(round(self.p["cover"]))))
        want = forecast[:, :weeks].mean(axis=1) * self.p["cover"]
        stock = seen(observation, "stock.qty")[self.stock_row]
        backlog = seen(observation, "backlog.qty")
        live = observation["pipeline.qty.observed"] == 1
        head, k, qty = (
            self.edge_head[observation["pipeline.edge"]],
            observation["pipeline.k"],
            observation["pipeline.qty"],
        )
        for r, (node, comm) in enumerate(self.demands):
            idx = self.members[r]
            if idx.size == 0:
                continue
            transit = qty[live & (head == node) & (k == comm)].sum()
            limit = max(0.0, self.p["slack"] * (want[r] + backlog[r] - stock[r] - transit))
            total = plan.flows[idx].sum()
            if total > limit:
                plan.flows[idx] *= limit / total
        return plan
