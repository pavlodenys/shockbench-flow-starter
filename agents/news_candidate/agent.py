"""Allocate fuel using inventory, inbound cargo, travel times and expected grid use.

A small LP balances fuel coverage by arrival week. Semiconductor flows retain the template
policy. Only public configuration and observations are read; imports are submission-safe.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


PARAMS = {"reserve_weeks": 3.0, "discount": 0.85, "allocation_blend": 0.5}
_params_file = Path(__file__).with_name("params.json")
if _params_file.is_file():
    PARAMS.update(json.loads(_params_file.read_text()))


class Agent:
    def __init__(self, config):
        self.T = int(config["T"])
        self.reserve = float(PARAMS["reserve_weeks"])
        self.discount = float(PARAMS["discount"])
        self.blend = float(PARAMS["allocation_blend"])
        if self.reserve < 0 or not 0 < self.discount <= 1 or not 0 <= self.blend <= 1:
            raise ValueError("Invalid reserve, discount or allocation_blend")
        s, layout = config["static"], config["layout"]
        self.edges, self.lanes = s["edges"], s["lanes"]
        self.slot_edge = np.asarray(s["action_slots"]["edge"], dtype=int)
        self.slot_k = np.asarray(s["action_slots"]["k"], dtype=int)
        self.cap = np.asarray([self.edges["u0"][e] for e in self.slot_edge], dtype=float)
        self.stock_index = {tuple(p): i for i, p in enumerate(layout["stock_slots"])}
        self.chk_pos = {n: i for i, n in enumerate(layout["chokepoints"])}
        self.lot_keys = layout.get("lot_keys")
        self.pool = s["commodities"]["pool"]
        names = {name: i for i, name in enumerate(s["nodes"]["id"])}
        commodities = {name: i for i, name in enumerate(s["commodities"]["id"])}
        raw_nodes = {names[n["id"]]: n for n in s["instance"]["nodes"]}
        self.grid_nominal = np.array([raw_nodes[n]["grid"]["deliverable"] for n in layout["grids"]])
        self.grid_storage = {}
        self.groups, self.group_for = [], {}
        # A grid and directly connected terminals share a fuel inventory position.
        for gi, node in enumerate(layout["grids"]):
            grid = raw_nodes[node]["grid"]
            terminals = {
                self.edges["tail"][e]
                for e, head in enumerate(self.edges["head"])
                if head == node and s["nodes"]["type"][self.edges["tail"][e]] == "terminal"
            }
            for name, share in grid["shares"].items():
                if name not in commodities:
                    continue
                k, g = commodities[name], len(self.groups)
                members = {node, *terminals}
                for n in members:
                    self.group_for[n, k] = g
                indices = [self.stock_index[n, k] for n in members if (n, k) in self.stock_index]
                self.groups.append((gi, node, k, float(share), float(grid["voll"]), indices))
                self.grid_storage[self.stock_index[node, k]] = raw_nodes[node]["stock"][name]["storage"]
        self.routes, self.internal = [], []
        for slot, (e, k, lane) in enumerate(zip(self.slot_edge, self.slot_k, s["action_slots"]["lane"])):
            route = [int(e)] if lane is None else self.lanes["edges"][lane]
            dest, source = self.edges["head"][route[-1]], self.edges["tail"][e]
            group = self.group_for.get((dest, int(k)))
            if group is None:
                continue
            if self.group_for.get((source, int(k))) == group:
                self.internal.append((slot, group, self.stock_index[dest, int(k)]))
            else:
                self.routes.append((slot, group, self.stock_index[source, int(k)], route))
        self.nominal_tau = np.asarray(self.edges["tau0"], dtype=float)
        self.nominal_cost = np.asarray(self.edges["c0"], dtype=float)
        self.nominal_cap = np.asarray([x or 0.0 for x in self.edges["u0"]], dtype=float)
        self.value = np.asarray(s["commodities"]["v"], dtype=float)
        self.failures = 0
        self.news_weeks = 0
        self.nuclear_slots = np.array([s["commodities"]["id"][k] == "nucfuel" for k in self.slot_k])

    @staticmethod
    def _read(obs, key, fallback):
        return np.where(obs[key + ".observed"] == 1, obs[key], fallback)

    def _remaining(self, route, edge, tau, waits, k):
        rest = route[route.index(int(edge)) + 1 :]
        return sum(tau[e] + waits.get((self.edges["tail"][e], k), 0.0) for e in rest)

    def _news_reserve(self, obs, week, tau, waits, edge_cap):
        """Add at most three weeks of buffer for announced supply restrictions.

        Publications can be decoys. Do not treat them as current prohibitions or
        remember them after they disappear. Weight the buffer by the exposed
        share of currently available route capacity, without counting duplicates.
        """
        extra = np.zeros(len(self.groups))
        fields = ["pending_prohibitions." + name for name in ("edge", "k", "effective_week")]
        if any(key not in obs or key + ".observed" not in obs for key in fields):
            return extra
        visible = np.logical_and.reduce([np.asarray(obs[key + ".observed"]) == 1 for key in fields])
        pending = {}
        for i in np.flatnonzero(visible):
            edge, k, effective = (int(obs[key][i]) for key in fields)
            if week < effective <= min(self.T, week + 8):
                pending[edge, k] = min(pending.get((edge, k), effective), effective)
        if not pending:
            return extra
        total = np.zeros(len(self.groups))
        exposed = np.zeros(len(self.groups))
        for slot, group, source, route in self.routes:
            if not obs["action_mask"][slot] or self.nuclear_slots[slot]:
                continue
            capacity = min(self.cap[slot], max(0.0, edge_cap[self.slot_edge[slot]]))
            total[group] += capacity
            k = int(self.slot_k[slot])
            elapsed, affected, can_pass = 0.0, False, True
            for edge in route:
                elapsed += waits.get((self.edges["tail"][edge], k), 0.0)
                effective = pending.get((int(edge), k))
                if effective is not None:
                    affected = True
                    can_pass &= week + np.ceil(elapsed) < effective
                elapsed += tau[edge]
            # Only build ahead when a shipment can enter every threatened edge
            # before its stated deadline and reach the grid within the episode.
            dest = self.edges["head"][route[-1]]
            if affected and can_pass and elapsed + 1 + (dest != self.groups[group][1]) <= self.T - week + 1:
                exposed[group] += capacity
        np.divide(3.0 * exposed, total, out=extra, where=total > 0)
        return extra

    def act(self, observation):
        obs = observation
        flows = self.cap * obs["action_mask"]
        if not self.routes:
            return {"flows": flows}
        required = [i for group in self.groups for i in group[-1]] + [r[2] for r in self.routes]
        if not np.all(obs["stock.qty.observed"][required] == 1):
            return {"flows": flows}
        week = int(obs["week"][0])
        remaining = self.T - week + 1
        tau = self._read(obs, "graph_now.tau", self.nominal_tau)
        edge_cap = self._read(obs, "graph_now.u", self.nominal_cap)
        cost = self._read(obs, "graph_now.c", self.nominal_cost)
        tariff = self._read(obs, "graph_now.tariff", 0.0)
        stock = np.asarray(obs["stock.qty"], dtype=float)
        generation = self._read(obs, "graph_now.grid.G_bar", self.grid_nominal)
        rate = np.array([share * generation[gi] for gi, node, k, share, voll, indices in self.groups])
        inventory = np.array([sum(stock[indices]) for *_, indices in self.groups])
        queues = []
        if self.lot_keys is not None:
            for key, qty, mask in zip(self.lot_keys, obs["queue_lots.qty"], obs["queue_lots.qty.observed"]):
                amount = float(np.sum(np.where(mask == 1, qty, 0.0)))
                if amount > 0:
                    queues.append((*key, amount))
        else:
            for i in np.flatnonzero(obs["queue_lots.qty.observed"]):
                queues.append(
                    (
                        int(obs["queue_lots.chokepoint"][i]),
                        int(obs["queue_lots.k"][i]),
                        int(obs["queue_lots.lane"][i]),
                        int(obs["queue_lots.next_edge"][i]),
                        float(obs["queue_lots.qty"][i]),
                    )
                )
        pool_qty = {}
        for node, k, lane, edge, qty in queues:
            key = node, self.pool[k]
            pool_qty[key] = pool_qty.get(key, 0.0) + qty
        waits = {}
        for node, pos in self.chk_pos.items():
            for k in {g[2] for g in self.groups}:
                key = "graph_now.kappa." + self.pool[k]
                capacity = float(obs[key][pos]) if obs[key + ".observed"][pos] else 0.0
                delay = pool_qty.get((node, self.pool[k]), 0.0) / max(capacity, 1e-9)
                waits[node, k] = min(12.0, delay) if capacity > 0 else 6.0
        news_reserve = self._news_reserve(obs, week, tau, waits, edge_cap)
        self.news_weeks += int(np.any(news_reserve > 0))
        arrivals = [[] for _ in self.groups]
        for i in np.flatnonzero(obs["pipeline.qty.observed"]):
            e, k = int(obs["pipeline.edge"][i]), int(obs["pipeline.k"][i])
            lane = int(obs["pipeline.lane"][i]) if obs["pipeline.lane.observed"][i] else None
            route = [e] if lane is None else self.lanes["edges"][lane]
            dest = self.edges["head"][route[-1]]
            group = self.group_for.get((dest, k))
            if group is not None:
                eta = float(obs["pipeline.arrival_week"][i]) - week + 1
                eta += self._remaining(route, e, tau, waits, k) + (dest != self.groups[group][1])
                arrivals[group].append((eta, float(obs["pipeline.qty"][i])))
        for node, k, lane, edge, qty in queues:
            route = self.lanes["edges"][lane]
            dest = self.edges["head"][route[-1]]
            group = self.group_for.get((dest, k))
            if group is not None:
                eta = 1 + waits[node, k] + tau[edge] + self._remaining(route, edge, tau, waits, k)
                arrivals[group].append((eta + (dest != self.groups[group][1]), qty))
        n = len(self.routes)
        eta, bounds, route_cost = [], [], []
        for slot, group, source, route in self.routes:
            k = self.slot_k[slot]
            dest = self.edges["head"][route[-1]]
            delay = sum(tau[e] for e in route) + 1 + (dest != self.groups[group][1])
            delay += sum(waits.get((self.edges["tail"][e], k), 0.0) for e in route)
            eta.append(delay)
            cap = min(self.cap[slot], max(0.0, edge_cap[self.slot_edge[slot]]), stock[source])
            bounds.append((0.0, cap if obs["action_mask"][slot] and delay <= remaining else 0.0))
            route_cost.append(sum(cost[e] + tariff[e, k] * self.value[k] for e in route) / 1e6)
        horizon = min(remaining, 20)
        rows, limits = [], []
        for source in sorted({r[2] for r in self.routes}):
            rows.append([float(r[2] == source) for r in self.routes])
            limits.append(stock[source])
        for edge in sorted({self.slot_edge[r[0]] for r in self.routes}):
            rows.append([float(self.slot_edge[r[0]] == edge) for r in self.routes])
            limits.append(max(0.0, edge_cap[edge]))
        for g in range(len(self.groups)):
            eligible = [eta[j] for j, r in enumerate(self.routes) if r[1] == g and bounds[j][1] > 0]
            cover = min(remaining, max(eligible, default=horizon) + self.reserve + news_reserve[g])
            incoming = sum(q for t, q in arrivals[g] if t <= cover)
            target = max(0.0, rate[g] * cover - inventory[g] - incoming)
            rows.append([float(r[1] == g) for r in self.routes])
            limits.append(target)
        shortage_rows, penalties = [], []
        for g, group in enumerate(self.groups):
            for t in range(1, horizon + 1):
                need = rate[g] * min(t + 1.0 + news_reserve[g], remaining) - inventory[g]
                # Cargo in a distant queue cannot cover this week's shortage.
                # Apply the same arrival cutoff to existing and new shipments.
                need -= sum(q for arrival, q in arrivals[g] if arrival <= t)
                shortage_rows.append([-float(r[1] == g and eta[j] <= t) for j, r in enumerate(self.routes)])
                limits.append(-need)
                penalties.append(group[4] / 1e6 * self.discount ** (t - 1))
        m = len(shortage_rows)
        a = np.zeros((len(rows) + m, n + m))
        a[: len(rows), :n] = rows
        a[len(rows) :, :n] = shortage_rows
        a[len(rows) :, n:] = -np.eye(m)
        result = linprog(
            route_cost + penalties,
            A_ub=a,
            b_ub=limits,
            bounds=bounds + [(0.0, None)] * m,
            method="highs",
            options={"time_limit": 0.25},
        )
        if result.success and np.all(np.isfinite(result.x)):
            for j, (slot, group, source, route) in enumerate(self.routes):
                # Keep part of the template request: current delays do not predict
                # future closures, and pipeline estimates can be optimistic.
                flows[slot] = (1 - self.blend) * flows[slot] + self.blend * max(0.0, min(result.x[j], bounds[j][1]))
        else:
            self.failures += 1
        # Avoid throwing delivered fuel away when the receiving grid is full.
        for slot, group, dest_stock in self.internal:
            capacity = self.grid_storage[dest_stock]
            flows[slot] = min(flows[slot], max(0.0, capacity + rate[group] - stock[dest_stock]))
        # Preserve long-term nuclear supply: lost shipments cannot be recovered
        # when the remaining supply capacity is below weekly consumption.
        flows[self.nuclear_slots] = self.cap[self.nuclear_slots] * obs["action_mask"][self.nuclear_slots]
        return {"flows": flows}

