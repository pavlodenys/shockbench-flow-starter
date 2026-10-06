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
        self.ration_floor = {}
        for g, (_, node, k, _, _, _) in enumerate(self.groups):
            grid = raw_nodes[node]["grid"]
            name = s["commodities"]["id"][k]
            if grid.get("rationed") == name:
                self.ration_floor[g] = s["instance"]["params"]["psi"] * grid["ibar"][name]
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
        self.reroute_weeks = 0
        self.reroute_requested_qty = 0.0
        self.nuclear_slots = np.array([s["commodities"]["id"][k] == "nucfuel" for k in self.slot_k])
        self.route_choices = {}
        self.slot_paths = [
            [int(edge)] if lane is None else self.lanes["edges"][lane]
            for edge, lane in zip(self.slot_edge, s["action_slots"]["lane"])
        ]
        for slot, group, source, route in self.routes:
            if not self.nuclear_slots[slot]:
                self.route_choices.setdefault((int(self.slot_edge[slot]), group), []).append((slot, route))

    @staticmethod
    def _read(obs, key, fallback):
        return np.where(obs[key + ".observed"] == 1, obs[key], fallback)

    def _remaining(self, route, edge, tau, waits, k):
        rest = route[route.index(int(edge)) + 1 :]
        return sum(tau[e] + waits.get((self.edges["tail"][e], k), 0.0) for e in rest)

    def _resources(self, route, k):
        """An edge and its chokepoint pool both constrain a shipment."""
        result = set(route)
        for edge in route:
            node = self.edges["tail"][edge]
            if node in self.chk_pos:
                result.add((node, self.pool[k]))
        return result

    def _routing_state(self, flows, obs, edge_cap):
        capacities = dict(enumerate(np.maximum(edge_cap, 0.0)))
        for node, pos in self.chk_pos.items():
            for pool in set(self.pool):
                key = "graph_now.kappa." + pool
                capacities[node, pool] = max(0.0, float(obs[key][pos])) if obs[key + ".observed"][pos] else 0.0
        committed, edge_queue = {}, {}

        def reserve(route, k, qty):
            for resource in self._resources(route, k):
                committed[resource] = committed.get(resource, 0.0) + qty

        if self.lot_keys is not None:
            lots = [
                (key, float(np.sum(np.where(mask == 1, qty, 0.0))))
                for key, qty, mask in zip(self.lot_keys, obs["queue_lots.qty"], obs["queue_lots.qty.observed"])
            ]
        else:
            fields = ["queue_lots." + key for key in ("chokepoint", "k", "lane", "next_edge", "qty")]
            visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
            lots = [
                ([int(obs[key][i]) for key in fields[:-1]], float(obs[fields[-1]][i])) for i in np.flatnonzero(visible)
            ]
        for (_, k, lane, edge), qty in lots:
            if qty <= 0:
                continue
            edge_queue[edge] = edge_queue.get(edge, 0.0) + qty
            route = self.lanes["edges"][lane]
            reserve(route[route.index(edge) :], k, qty)
        fields = ["pipeline." + key for key in ("edge", "k", "lane", "qty")]
        visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
        for i in np.flatnonzero(visible):
            edge, k, lane = (int(obs[key][i]) for key in fields[:-1])
            qty = float(obs[fields[-1]][i])
            if qty > 0 and lane >= 0:
                route = self.lanes["edges"][lane]
                reserve(route[route.index(edge) + 1 :], k, qty)

        # Estimate dispatch after entry-edge sharing. Source stocks and fleet
        # limits can reduce it further; reserve this conservative upper bound.
        scale = np.ones(len(flows))
        entry_total = {}
        for slot, qty in enumerate(flows):
            edge = int(self.slot_edge[slot])
            entry_total[edge] = entry_total.get(edge, 0.0) + qty
        for slot in range(len(flows)):
            edge = int(self.slot_edge[slot])
            scale[slot] = min(scale[slot], capacities[edge] / max(entry_total[edge], 1e-9))
        loads = {}
        for slot, route in enumerate(self.slot_paths):
            for resource in self._resources(route, int(self.slot_k[slot])):
                loads[resource] = loads.get(resource, 0.0) + flows[slot] * scale[slot]
        return capacities, committed, edge_queue, scale, loads

    def _reroute(self, flows, obs, week, tau, waits, edge_cap, cost, tariff):
        """Shift requests between lanes with identical entry edge and fuel sink.

        Preserve their aggregate request (and therefore entry-edge/source load).
        Avoid a full switch because notices may be decoys and queues can change.
        Reserve downstream edge and pool capacity before adding another shipment.
        """
        fields = ["pending_prohibitions." + key for key in ("edge", "k", "effective_week")]
        pending = {}
        if all(key in obs and key + ".observed" in obs for key in fields):
            visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
            for i in np.flatnonzero(visible):
                edge, k, effective = (int(obs[key][i]) for key in fields)
                if week < effective <= self.T:
                    pending[edge, k] = min(pending.get((edge, k), effective), effective)
        capacities, committed, edge_queue, scale, loads = self._routing_state(flows, obs, edge_cap)
        moved = 0.0
        for (entry, group), choices in self.route_choices.items():
            if len(choices) < 2 or edge_cap[entry] <= 0:
                continue
            options = []
            resources, headroom = {}, {}
            for slot, route in choices:
                if not obs["action_mask"][slot]:
                    continue
                k = int(self.slot_k[slot])
                elapsed, queue_delay, risk = 0.0, 0.0, False
                resources[slot] = self._resources(route, k)
                headroom[slot] = {}
                travel = 0.0
                for edge in route:
                    node = self.edges["tail"][edge]
                    edge_wait = edge_queue.get(edge, 0.0) / max(edge_cap[edge], 1e-9)
                    wait = max(waits.get((node, k), 0.0), edge_wait)
                    # Count service only until nominal arrival, plus this week's
                    # service. A long queue must not create extra apparent room.
                    for resource in self._resources([edge], k):
                        room = capacities[resource] * (travel + 1) - committed.get(resource, 0.0)
                        headroom[slot][resource] = min(headroom[slot].get(resource, np.inf), room)
                    elapsed += wait
                    queue_delay += wait
                    if node in self.chk_pos:
                        pos = self.chk_pos[node]
                        if obs["graph_now.open.observed"][pos] and obs["graph_now.open"][pos] <= 0:
                            risk = True
                    if edge_cap[edge] <= 0 or week + np.ceil(elapsed) >= pending.get((int(edge), k), np.inf):
                        risk = True
                    elapsed += tau[edge]
                    travel += tau[edge]
                dest = self.edges["head"][route[-1]]
                eta = elapsed + 1 + (dest != self.groups[group][1])
                risk |= eta > self.T - week + 1
                price = sum(cost[e] + tariff[e, k] * self.value[k] for e in route)
                options.append((slot, bool(risk), eta, queue_delay, price))
            safe = sorted((x for x in options if not x[1]), key=lambda x: (x[2], x[4], x[0]))
            # Read donors from the original request so transfers do not cascade.
            original = flows.copy()
            for slot, risk, eta, queue_delay, price in options:
                allowance = 0.5 * original[slot]
                for target, _, target_eta, target_wait, target_price in safe:
                    if target == slot or target_price > 1.5 * max(price, 1e-9):
                        continue
                    delayed = queue_delay - target_wait >= 2 and eta - target_eta >= 2
                    if not (risk or delayed):
                        continue
                    shift = min(allowance, max(0.0, self.cap[target] - flows[target]))
                    if scale[target] <= 0:
                        continue
                    # Shared resources retain the same total request. For new
                    # resources, account for every other route's reservations.
                    for resource in resources[target] - resources[slot]:
                        room = headroom[target][resource] - loads.get(resource, 0.0)
                        shift = min(shift, max(0.0, room) / scale[target])
                    if shift <= 0:
                        continue
                    for resource in resources[slot]:
                        loads[resource] -= shift * scale[slot]
                    for resource in resources[target]:
                        loads[resource] += shift * scale[target]
                    flows[slot] -= shift
                    flows[target] += shift
                    allowance -= shift
                    moved += shift
                    if allowance <= 1e-9:
                        break
        self.reroute_weeks += int(moved > 0)
        self.reroute_requested_qty += moved
        return flows

    def _grid_supply(self, g, stock, events, edge_cap, tau, mask, horizon):
        """Cumulative fuel available at the grid, keeping terminal stock separate.

        Terminal dispatch uses start-of-week stock. Arrivals at a terminal can
        be forwarded next week; in-transit grid deliveries arrive directly.
        Current capacities are a forecast, not knowledge of future shocks.
        """
        _, grid, k, _, _, indices = self.groups[g]
        grid_index = self.stock_index[grid, k]
        balance = {i: float(stock[i]) for i in indices}
        deliveries = np.zeros(horizon + 1)
        due = {}
        for t, node, qty in events:
            due.setdefault(max(1, int(np.ceil(t))), []).append((node, qty))
        for t in range(1, horizon + 1):
            for slot, group, _ in self.internal:
                if group != g or not mask[slot]:
                    continue
                e = int(self.slot_edge[slot])
                source = self.stock_index[self.edges["tail"][e], k]
                # The shared edge also carries other fuels' template requests.
                edge_request = sum(self.cap[s] * mask[s] for s in np.flatnonzero(self.slot_edge == e))
                cap = min(self.cap[slot], max(0.0, edge_cap[e]) * self.cap[slot] / max(edge_request, 1e-9))
                q = min(balance[source], cap)
                balance[source] -= q
                due.setdefault(t + max(1, int(tau[e])), []).append((grid, q))
            for node, qty in due.get(t, []):
                index = self.stock_index.get((node, k))
                if index in balance:
                    balance[index] += qty
            deliveries[t] = balance[grid_index]
        return deliveries

    def _lng_curves(self, obs, week, stock, tau, waits, edge_cap, queues, bounds):
        horizon = self.T - week + 1
        events = {g: [] for g in self.ration_floor}
        fields = ["pipeline." + key for key in ("edge", "k", "qty", "arrival_week")]
        visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
        for i in np.flatnonzero(visible):
            e, k = int(obs["pipeline.edge"][i]), int(obs["pipeline.k"][i])
            lane = int(obs["pipeline.lane"][i]) if obs["pipeline.lane.observed"][i] else -1
            route = self.lanes["edges"][lane] if lane >= 0 else [e]
            dest = self.edges["head"][route[-1]]
            g = self.group_for.get((dest, k))
            if g in events:
                t = float(obs["pipeline.arrival_week"][i]) - week + 1
                t += self._remaining(route, e, tau, waits, k)
                events[g].append((t, dest, float(obs["pipeline.qty"][i])))
        for node, k, lane, e, q in queues:
            route = self.lanes["edges"][lane]
            dest = self.edges["head"][route[-1]]
            g = self.group_for.get((dest, k))
            if g in events:
                t = 1 + waits[node, k] + tau[e] + self._remaining(route, e, tau, waits, k)
                events[g].append((t, dest, q))
        supply, fractions = {}, {}
        for g in events:
            supply[g] = self._grid_supply(g, stock, events[g], edge_cap, tau, obs["action_mask"], horizon)
        for j, (slot, g, _, route) in enumerate(self.routes):
            if g not in events or bounds[j][1] <= 0:
                continue
            k = int(self.slot_k[slot])
            t = 1 + sum(tau[e] + waits.get((self.edges["tail"][e], k), 0.0) for e in route)
            extra = (t, self.edges["head"][route[-1]], bounds[j][1])
            curve = self._grid_supply(g, stock, events[g] + [extra], edge_cap, tau, obs["action_mask"], horizon)
            fractions[j] = np.clip((curve - supply[g]) / bounds[j][1], 0.0, 1.0)
        return supply, fractions

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
        grid_supply, grid_fraction = self._lng_curves(obs, week, stock, tau, waits, edge_cap, queues, bounds)
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
            cover = min(remaining, max(eligible, default=horizon) + self.reserve)
            incoming = sum(q for t, q in arrivals[g] if t <= cover)
            target = max(0.0, rate[g] * cover + self.ration_floor.get(g, 0.0) - inventory[g] - incoming)
            rows.append([float(r[1] == g) for r in self.routes])
            limits.append(target)
        shortage_rows, penalties = [], []
        for g, group in enumerate(self.groups):
            for t in range(1, horizon + 1):
                need = rate[g] * min(t + 1.0, remaining) - inventory[g]
                # Cargo in a distant queue cannot cover this week's shortage.
                # Apply the same arrival cutoff to existing and new shipments.
                need -= sum(q for arrival, q in arrivals[g] if arrival <= t)
                shortage_rows.append([-float(r[1] == g and eta[j] <= t) for j, r in enumerate(self.routes)])
                if g in grid_supply:
                    # End-of-week stock must also support next week's rationing.
                    floor = self.ration_floor[g] if t < remaining else 0.0
                    need = rate[g] * t + floor - grid_supply[g][t]
                    shortage_rows[-1] = [
                        -float(grid_fraction[j][t]) if r[1] == g and j in grid_fraction else 0.0
                        for j, r in enumerate(self.routes)
                    ]
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
        # Restore it before reserving shared capacity for routing decisions.
        flows[self.nuclear_slots] = self.cap[self.nuclear_slots] * obs["action_mask"][self.nuclear_slots]
        flows = self._reroute(flows, obs, week, tau, waits, edge_cap, cost, tariff)
        return {"flows": flows}
