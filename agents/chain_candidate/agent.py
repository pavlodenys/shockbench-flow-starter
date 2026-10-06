"""Allocate fuel using inventory, inbound cargo, travel times and expected grid use.

A small LP balances fuel coverage by arrival week. Semiconductor flows retain the template
policy. Only public configuration and observations are read; imports are submission-safe.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


PARAMS = {
    "reserve_weeks": 4.0,
    "discount": 0.9,
    "allocation_blend": 0.7,
    "planning_horizon": 20.0,
    "mpc_blend": 0.5,
    "eta_margin": 1.0,
    "supply_factor": 0.9,
    "demand_factor": 1.0,
    "terminal_weeks": 2.0,
    "terminal_weight": 0.1,
    "change_penalty": 0.02,
}
_params_file = Path(__file__).with_name("params.json")
if _params_file.is_file():
    PARAMS.update(json.loads(_params_file.read_text()))


class BaseAgent:
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

        self.slot_lane = s["action_slots"]["lane"]
        params = s["instance"]["params"]
        self.fleet_cap = {p: params["fleet_share"][p] * params["fleet_measure"][p] for p in set(self.pool)}
        self.nominal_kappa = {
            (n, p): raw_nodes[n]["chokepoint"]["mu"][p] * raw_nodes[n]["chokepoint"]["k_c"]
            for n in self.chk_pos
            for p in set(self.pool)
        }
        self.dup = set()
        edge_ids = {v: i for i, v in enumerate(self.edges["id"])}
        lane_ids = {v: i for i, v in enumerate(self.lanes["id"])}
        raw_edges, raw_lanes = s["instance"]["edges"], s["instance"]["lanes"]

        def replaced(ref):
            return self.lanes["edges"][lane_ids[ref["lane"]]] if "lane" in ref else [edge_ids[ref["edge"]]]

        for e, raw in enumerate(raw_edges):
            if raw.get("alt_of") and raw["mode"] == "sea":
                dt = self.nominal_tau[e] - sum(self.nominal_tau[x] for x in replaced(raw["alt_of"]))
                self.dup.add((e, None, dt))
        for lane, raw in enumerate(raw_lanes):
            path = self.lanes["edges"][lane]
            ref = raw.get("alt_of")
            if not ref or any(raw_edges[e]["mode"] != "sea" for e in path):
                continue
            old = replaced(ref)
            diverge = (
                next(e for e in path if e not in old)
                if "lane" in ref
                else next(e for e in path if self.edges["tail"][e] == self.edges["tail"][old[0]] and e != old[0])
            )
            suffix = path if "lane" in ref else path[path.index(diverge) :]
            dt = sum(self.nominal_tau[e] for e in suffix) - sum(self.nominal_tau[e] for e in old)
            self.dup.add((diverge, lane if diverge == path[0] else None, dt))
        self.fleet_terms = {}
        for e, lane, dt in self.dup:
            self.fleet_terms.setdefault(e, []).append((lane, dt))
        self._forecast_eta = {}

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
                eta = self._forecast_eta.get(slot, eta)
                queue_delay = max(0.0, eta - travel - 1 - (dest != self.groups[group][1]))
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

    def _forecast(self, obs, week, edge_cap, tau, bounds):
        """Release visible FIFO cohorts in weekly portions under current capacity.

        Forecast only observed cargo plus one simultaneous dispatch. Known
        closure ends restore nominal capacity; unknown future shocks and later
        decisions are not available. Fleet slack applies after edge/pool release.
        """
        horizon = min(self.T - week + 1, 32)
        arrivals = [[] for _ in self.groups]
        self._node_arrivals = []
        self._chain_arrivals = []
        self._chain_existing_use = {}
        delivered = np.zeros((len(self.routes), horizon + 1))
        route_j = {r[0]: j for j, r in enumerate(self.routes)}
        events, book = {}, []

        # Lot: cohort week, commodity, lane, next edge, quantity, candidate tag.
        def finish(t, k, lane, edge, q, tag):
            path = self.lanes["edges"][lane] if lane >= 0 else [edge]
            pos = path.index(edge)
            if pos + 1 < len(path):
                events.setdefault(t, []).append([t, k, lane, path[pos + 1], q, tag])
            else:
                dest = self.edges["head"][edge]
                if tag == -1:
                    self._chain_arrivals.append((t - week, dest, k, q))
                g = self.group_for.get((dest, k))
                if g is not None:
                    eta = t - week + 1 + (dest != self.groups[g][1])
                    if tag == -1:
                        self._node_arrivals.append((t - week + 1, dest, k, q))
                        arrivals[g].append((eta, q))
                    elif tag >= 0 and 0 <= eta <= horizon:
                        delivered[tag, int(eta)] += q

        if self.lot_keys is not None:
            for (node, k, lane, e), quantities, masks in zip(
                self.lot_keys, obs["queue_lots.qty"], obs["queue_lots.qty.observed"]
            ):
                for index in np.flatnonzero((masks == 1) & (quantities > 1e-8)):
                    book.append([int(index) + 1, k, lane, e, float(quantities[index]), -1])
        else:
            fields = ["queue_lots." + key for key in ("arrival_week", "k", "lane", "next_edge", "qty")]
            visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
            for i in np.flatnonzero(visible):
                if obs[fields[-1]][i] > 1e-8:
                    book.append([*(int(obs[f][i]) for f in fields[:-1]), float(obs[fields[-1]][i]), -1])
        fields = ["pipeline." + key for key in ("arrival_week", "k", "edge", "qty")]
        visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in fields])
        for i in np.flatnonzero(visible):
            t, k, e = (int(obs[f][i]) for f in fields[:-1])
            lane = int(obs["pipeline.lane"][i]) if obs["pipeline.lane.observed"][i] else -1
            if obs[fields[-1]][i] > 1e-8:
                finish(max(week, t), k, lane, e, float(obs[fields[-1]][i]), -1)
        reopening = {}
        fields = ["closure_end." + k for k in ("chokepoint", "end_week")]
        if all(f in obs for f in fields):
            visible = np.logical_and.reduce([obs[f + ".observed"] == 1 for f in fields])
            for i in np.flatnonzero(visible):
                node, end = (int(obs[f][i]) for f in fields)
                reopening[node] = max(reopening.get(node, week), end + 1)
        prohibited = self._read(obs, "graph_now.prohibited", 1).astype(bool)
        pending = {}
        fields = ["pending_prohibitions." + k for k in ("edge", "k", "effective_week")]
        if all(f in obs for f in fields):
            visible = np.logical_and.reduce([obs[f + ".observed"] == 1 for f in fields])
            for i in np.flatnonzero(visible):
                e, k, effective = (int(obs[f][i]) for f in fields)
                pending[e, k] = min(pending.get((e, k), self.T + 1), effective)
        # Reproduce entry-edge and stock sharing for this week's template load.
        request = self.cap * obs["action_mask"]
        for e in set(self.slot_edge):
            idx = np.flatnonzero(self.slot_edge == e)
            request[idx] *= min(1.0, max(0.0, edge_cap[e]) / max(request[idx].sum(), 1e-9))
        sources = [(self.edges["tail"][e], int(k)) for e, k in zip(self.slot_edge, self.slot_k)]
        for source in set(sources):
            idx = [s for s, pair in enumerate(sources) if pair == source]
            si = self.stock_index.get(source)
            available = obs["stock.qty"][si] if si is not None and obs["stock.qty.observed"][si] else 0.0
            request[idx] *= min(1.0, available / max(request[idx].sum(), 1e-9))
        for offset in range(horizon):
            t = week + offset
            book.extend(events.pop(t, []))
            caps = np.maximum(edge_cap, 0.0).copy()
            pool_caps = {}
            for node, ci in self.chk_pos.items():
                reopen = t >= reopening.get(node, self.T + 1)
                for pool in set(self.pool):
                    key = "graph_now.kappa." + pool
                    pool_caps[node, pool] = (
                        self.nominal_kappa[node, pool]
                        if reopen
                        else float(obs[key][ci])
                        if obs[key + ".observed"][ci]
                        else 0.0
                    )
                if reopen:
                    for e, tail in enumerate(self.edges["tail"]):
                        if tail == node:
                            caps[e] = self.nominal_cap[e]
            cohorts = {}
            for lot in book:
                if lot[4] > 1e-8:
                    cohorts.setdefault((self.edges["tail"][lot[3]], lot[0], self.pool[lot[1]]), []).append(lot)
            releases = []
            for (node, _, pool), lots in sorted(cohorts.items()):
                amounts = [
                    0.0 if prohibited[lt[3], lt[1]] or t >= pending.get((lt[3], lt[1]), self.T + 1) else lt[4]
                    for lt in lots
                ]
                totals = {}
                for lt, q in zip(lots, amounts):
                    totals[lt[3]] = totals.get(lt[3], 0.0) + q
                amounts = [q * min(1.0, caps[lt[3]] / max(totals[lt[3]], 1e-9)) for lt, q in zip(lots, amounts)]
                scale = min(1.0, max(0.0, pool_caps[node, pool]) / max(sum(amounts), 1e-9))
                for lt, q in zip(lots, amounts):
                    q *= scale
                    caps[lt[3]] = max(0.0, caps[lt[3]] - q)
                    pool_caps[node, pool] -= q
                    if q > 1e-8:
                        releases.append((lt, q))
            dispatch = []
            if offset == 0:
                for slot, q in enumerate(request):
                    if q > 1e-8:
                        lane = self.slot_lane[slot]
                        dispatch.append(
                            (
                                [
                                    t,
                                    int(self.slot_k[slot]),
                                    -1 if lane is None else lane,
                                    int(self.slot_edge[slot]),
                                    q,
                                    route_j.get(slot, -2),
                                ],
                                q,
                            )
                        )
            all_moves = dispatch + releases
            fleet_totals = {p: 0.0 for p in set(self.pool)}
            charges = []
            for lot, q in all_moves:
                dt = sum(d for ln, d in self.fleet_terms.get(lot[3], []) if ln is None or ln == lot[2])
                charges.append(dt)
                fleet_totals[self.pool[lot[1]]] += dt * q
            for (lot, q), dt in zip(all_moves, charges):
                if dt > 0:
                    p = self.pool[lot[1]]
                    q *= min(1.0, self.fleet_cap[p] / max(fleet_totals[p], 1e-9))
                if lot[5] == -1:
                    edge, commodity = lot[3], lot[1]
                    resources = [edge]
                    if self.edges["tail"][edge] in self.chk_pos:
                        resources.append((self.edges["tail"][edge], self.pool[commodity]))
                    for resource in resources:
                        key = (t - week, resource)
                        self._chain_existing_use[key] = self._chain_existing_use.get(key, 0.0) + q
                lot[4] -= q
                finish(t + max(1, int(tau[lot[3]])), lot[1], lot[2], lot[3], q, lot[5])
            book = [lot for lot in book if lot[4] > 1e-8]
        fractions = np.cumsum(delivered, axis=1)
        etas = []
        for j, (slot, _, _, _) in enumerate(self.routes):
            cap = bounds[j][1]
            if cap > 0:
                fractions[j] = np.minimum(1.0, fractions[j] / cap)
            else:
                fractions[j] = 0.0
            q = delivered[j].sum()
            eta = float(np.dot(np.arange(horizon + 1), delivered[j]) / q) if q > 1e-8 else self.T + 1.0
            etas.append(eta)
        self._forecast_eta = {r[0]: etas[j] for j, r in enumerate(self.routes)}
        return arrivals, fractions, etas

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
        n = len(self.routes)
        bounds, route_cost = [], []
        for slot, group, source, route in self.routes:
            k = self.slot_k[slot]
            cap = min(self.cap[slot], max(0.0, edge_cap[self.slot_edge[slot]]), stock[source])
            bounds.append((0.0, cap if obs["action_mask"][slot] else 0.0))
            route_cost.append(sum(cost[e] + tariff[e, k] * self.value[k] for e in route) / 1e6)
        arrivals, arrival_fraction, eta = self._forecast(obs, week, edge_cap, tau, bounds)
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
            target = max(0.0, rate[g] * cover - inventory[g] - incoming)
            rows.append([float(r[1] == g) for r in self.routes])
            limits.append(target)
        shortage_rows, penalties = [], []
        for g, group in enumerate(self.groups):
            for t in range(1, horizon + 1):
                need = rate[g] * min(t + 1.0, remaining) - inventory[g]
                # Cargo in a distant queue cannot cover this week's shortage.
                # Apply the same arrival cutoff to existing and new shipments.
                need -= sum(q for arrival, q in arrivals[g] if arrival <= t)
                shortage_rows.append(
                    [-float(arrival_fraction[j, t]) if r[1] == g else 0.0 for j, r in enumerate(self.routes)]
                )
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


class FuelAgent(BaseAgent):
    """Receding-horizon fuel plan with separate terminal/grid inventory balances.

    Future supply/capacity are approximations of public current observations.
    Nuclear requests retain v8 behavior. Final goods avoid nearly closed straits.
    Execute only step zero of the fuel plan.
    """

    def __init__(self, config):
        super().__init__(config)
        self.options = dict(PARAMS)
        node_types = config["static"]["nodes"]["type"]
        lane_chokepoints = config["static"]["lanes"]["chokepoints"]
        self.final_route_chokepoints = [
            [self.chk_pos[node] for node in lane_chokepoints[lane]]
            if lane is not None and node_types[self.edges["head"][self.slot_paths[slot][-1]]] == "sink"
            else []
            for slot, lane in enumerate(self.slot_lane)
        ]
        self.mpc_failures = 0
        self.mpc_solves = 0
        self._supply_pairs = [tuple(pair) for pair in config["layout"]["supply_slots"]]
        raw = config["static"]["instance"]["nodes"]
        names = config["static"]["commodities"]["id"]
        self._stock_limits = np.array(
            [raw[n]["stock"][names[k]]["storage"] for n, k in config["layout"]["stock_slots"]], dtype=float
        )
        self._holding = np.array(
            [raw[n]["stock"][names[k]].get("holding_cost", 0.0) for n, k in config["layout"]["stock_slots"]],
            dtype=float,
        )

    def act(self, observation):
        baseline = super().act(observation)["flows"]
        # A near-closed strait should not take OSAT stock from an open route to market.
        open_now = observation["graph_now.open"]
        seen = observation["graph_now.open.observed"]
        for slot, chokepoints in enumerate(self.final_route_chokepoints):
            for pos in chokepoints:
                if seen[pos]:
                    baseline[slot] *= max(0.0, min(1.0, float(open_now[pos]))) ** 2.0
        if self.options["mpc_blend"] == 0 or not self.routes:
            return {"flows": baseline}
        planned = self._plan(observation, baseline)
        if planned is None:
            self.mpc_failures += 1
            return {"flows": baseline}
        blend = self.options["mpc_blend"]
        flows = baseline.copy()
        for slot, qty in planned.items():
            flows[slot] = (1 - blend) * baseline[slot] + blend * qty
        return {"flows": flows}

    def _plan(self, obs, baseline):
        H = min(int(round(self.options["planning_horizon"])), self.T - int(obs["week"][0]) + 1)
        if H < 1:
            return None
        # External routes plus terminal-to-grid edges; nuclear is held fixed.
        routes = [
            (slot, group, source, route) for slot, group, source, route in self.routes if not self.nuclear_slots[slot]
        ]
        routes += [
            (
                slot,
                group,
                self.stock_index[self.edges["tail"][self.slot_edge[slot]], int(self.slot_k[slot])],
                [int(self.slot_edge[slot])],
            )
            for slot, group, _ in self.internal
            if not self.nuclear_slots[slot]
        ]
        if not routes:
            return {}
        slots = {r[0] for r in routes}
        pairs = sorted(
            {(self.edges["tail"][r[3][0]], int(self.slot_k[r[0]])) for r in routes}
            | {(self.edges["head"][r[3][-1]], int(self.slot_k[r[0]])) for r in routes}
        )
        if any(obs["stock.qty.observed"][self.stock_index[p]] != 1 for p in pairs):
            return None
        pair_pos = {pair: i for i, pair in enumerate(pairs)}
        R, N = len(routes), len(pairs)
        group_at = {(g[1], g[2]): index for index, g in enumerate(self.groups)}
        cap = np.maximum(0, self._read(obs, "graph_now.u", self.nominal_cap))
        tau = self._read(obs, "graph_now.tau", self.nominal_tau)
        freight = self._read(obs, "graph_now.c", self.nominal_cost)
        tariff = self._read(obs, "graph_now.tariff", 0.0)
        generation = self._read(obs, "graph_now.grid.G_bar", self.grid_nominal)
        supply = self._read(obs, "graph_now.supply.avail", 0.0)
        resource_caps, _, _, _, _ = self._routing_state(baseline, obs, cap)
        protected_load = {}
        for slot, qty in enumerate(baseline):
            if slot not in slots:
                for resource in self._resources(self.slot_paths[slot], int(self.slot_k[slot])):
                    protected_load[resource] = protected_load.get(resource, 0.0) + qty
        incoming = np.zeros((H, N))
        for eta, node, k, qty in self._node_arrivals:
            pair = (node, k)
            t = max(0, int(np.ceil(eta)) - 1)
            if pair in pair_pos and t < H:
                incoming[t, pair_pos[pair]] += qty
        # x[t,route], inventory[t,node], unmet burn[t,node], disposal[t,node], deviations.
        INV, U, W, D = H * R, H * (R + N), H * (R + 2 * N), H * (R + 3 * N)
        size = D + R
        objective = np.zeros(size)
        bounds = [(0.0, None)] * size
        eq_rows, eq_cols, eq_values, rhs = [], [], [], []
        ub_rows, ub_cols, ub_values, ub_rhs = [], [], [], []

        def eq(row, col, val):
            eq_rows.append(row)
            eq_cols.append(col)
            eq_values.append(val)

        def ub(terms, limit):
            row = len(ub_rhs)
            for col, val in terms:
                ub_rows.append(row)
                ub_cols.append(col)
                ub_values.append(val)
            ub_rhs.append(max(0.0, float(limit)))

        initial = np.array([obs["stock.qty"][self.stock_index[p]] for p in pairs])
        # Reserve v8's other flows on shared edges and stock before allocating.
        other_edge, other_stock = {}, {}
        for slot, q in enumerate(baseline):
            if slot in slots:
                continue
            e, k = int(self.slot_edge[slot]), int(self.slot_k[slot])
            other_edge[e] = other_edge.get(e, 0.0) + q
            p = (self.edges["tail"][e], k)
            other_stock[p] = other_stock.get(p, 0.0) + q
        supply_at = {p: max(0.0, supply[j]) * self.options["supply_factor"] for j, p in enumerate(self._supply_pairs)}
        rates = np.zeros(N)
        for i, pair in enumerate(pairs):
            g = group_at.get(pair)
            if g is not None:
                group = self.groups[g]
                rates[i] = group[3] * generation[group[0]] * self.options["demand_factor"]
        for t in range(H):
            weight = self.discount**t
            for i, pair in enumerate(pairs):
                row = t * N + i
                s = self.stock_index[pair]
                eq(row, INV + row, 1.0)
                if t:
                    eq(row, INV + row - N, -1.0)
                eq(row, U + row, -1.0)
                eq(row, W + row, 1.0)
                rhs.append((initial[i] if t == 0 else 0.0) + incoming[t, i] + supply_at.get(pair, 0.0) - rates[i])
                bounds[INV + row] = (0.0, self._stock_limits[s])
                bounds[U + row] = (0.0, rates[i])
                g = group_at.get(pair)
                penalty = self.groups[g][4] / 1e6 if g is not None else 0.0
                objective[U + row] = penalty * weight
                objective[W + row] = (0.0 if pair in supply_at else max(penalty, 1.0)) * weight
                objective[INV + row] = self._holding[s] / 1e6 * weight
                # Bounded terminal stock value prevents exhausting the planning horizon.
                if t == H - 1 and g is not None:
                    target = min(self._stock_limits[s], rates[i] * self.options["terminal_weeks"])
                    # Extra stock above target has no speculative terminal reward.
                    bounds[INV + row] = (0.0, self._stock_limits[s])
                    col = len(objective)
                    objective = np.append(objective, self.options["terminal_weight"] * penalty)
                    bounds.append((0.0, target))
                    ub([(INV + row, -1.0), (col, -1.0)], -target)
                    # ub helper clamps resource limits; restore this negative target.
                    ub_rhs[-1] = -target
            for j, (slot, g, source, route) in enumerate(routes):
                k = int(self.slot_k[slot])
                col = t * R + j
                source_pair = (self.edges["tail"][route[0]], k)
                dest_pair = (self.edges["head"][route[-1]], k)
                si, di = pair_pos[source_pair], pair_pos[dest_pair]
                eq(t * N + si, col, 1.0)
                eta = self._forecast_eta.get(slot, sum(tau[e] for e in route))
                if len(route) == 1 and source_pair in group_at:
                    eta = tau[route[0]]
                # Explicit transfer means terminal arrivals no longer include grid's +1 week.
                if slot in self._forecast_eta and dest_pair not in group_at:
                    eta -= 1
                delay = max(1, int(np.ceil(eta + self.options["eta_margin"])))
                arrival = t + delay - 1
                if arrival < H:
                    eq(arrival * N + di, col, -1.0)
                limit = min(self.cap[slot], min(cap[e] for e in route)) if obs["action_mask"][slot] else 0.0
                if arrival >= H:
                    limit = 0.0
                bounds[col] = (0.0, max(0.0, limit))
                objective[col] = weight * sum(freight[e] + tariff[e, k] * self.value[k] for e in route) / 1e6
                if t == 0:
                    objective[D + j] = self.options["change_penalty"] * self.groups[g][4] / 1e6
                    ub([(col, 1.0), (D + j, -1.0)], baseline[slot])
                    ub([(col, -1.0), (D + j, -1.0)], -baseline[slot])
                    ub_rhs[-1] = -baseline[slot]
            for pair in pairs:
                i = pair_pos[pair]
                terms = [
                    (t * R + j, 1.0)
                    for j, r in enumerate(routes)
                    if (self.edges["tail"][r[3][0]], int(self.slot_k[r[0]])) == pair
                ]
                if t:
                    terms.append((INV + (t - 1) * N + i, -1.0))
                ub(terms, initial[i] - other_stock.get(pair, 0.0) if t == 0 else 0.0)
            # Conservative route throughput: share every downstream edge/pool,
            # not just entry capacity. ETA accounts for visible queued cargo.
            resources = set(
                resource for slot, _, _, route in routes for resource in self._resources(route, int(self.slot_k[slot]))
            )
            for resource in resources:
                ub(
                    [
                        (t * R + j, 1.0)
                        for j, (slot, _, _, route) in enumerate(routes)
                        if resource in self._resources(route, int(self.slot_k[slot]))
                    ],
                    resource_caps[resource] - protected_load.get(resource, 0.0),
                )
        matrix = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(H * N, len(objective))).tocsr()
        inequalities = coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(len(ub_rhs), len(objective))).tocsr()
        result = linprog(
            objective,
            A_eq=matrix,
            b_eq=rhs,
            A_ub=inequalities,
            b_ub=ub_rhs,
            bounds=bounds,
            method="highs",
            options={"time_limit": 0.45},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        self.mpc_solves += 1
        return {
            slot: (
                min(baseline[slot], obs["stock.qty"][source])
                if self._forecast_eta.get(slot, 0.0) + self.options["eta_margin"] > H
                else max(0.0, min(result.x[j], self.cap[slot]))
            )
            for j, (slot, _, source, _) in enumerate(routes)
        }


class Agent(FuelAgent):
    """Experimental public-observation MPC across fuel, fabs, OSAT and demand.

    Production is automatic in the simulator. Fix its first-week rules and
    relax future starts, then replan every week. Nuclear actions stay at the
    established fuel policy. Unknown future disruptions are not available.
    """

    def __init__(self, config):
        super().__init__(config)
        self.layout = config["layout"]
        self.static = config["static"]
        self.raw_nodes = self.static["instance"]["nodes"]
        self.goods = self.static["commodities"]["id"]
        self.names = self.static["nodes"]["id"]
        self.node_by_name = {name: i for i, name in enumerate(self.names)}
        self.pairs = [tuple(p) for p in self.layout["stock_slots"]]
        self.fab_attrs = [self.raw_nodes[n]["fab"] for n in self.layout["fabs"]]
        self.grid_attrs = [self.raw_nodes[n]["grid"] for n in self.layout["grids"]]
        self.osat_attrs = [self.raw_nodes[n]["osat"] for n in self.layout["osats"]]
        self.grid_members = [
            [i for i, f in enumerate(self.fab_attrs) if f.get("grid") == self.names[n]] for n in self.layout["grids"]
        ]
        self.chip_slots = np.array([self.pool[k] == "ct" for k in self.slot_k])
        self.chain_solves = self.chain_failures = 0
        self.chain_horizon = int(PARAMS.get("chain_horizon", 24))
        self.chip_blend = float(PARAMS.get("chip_blend", 0.8))
        self.fuel_blend = float(PARAMS.get("fuel_blend", 0.0))
        self.last_chain_diagnostics = {}

    def act(self, obs):
        baseline = super().act(obs)["flows"]
        planned = self._chain_plan(obs, baseline)
        if planned is None:
            self.chain_failures += 1
            return {"flows": baseline}
        flows = baseline.copy()
        for slot, qty in enumerate(planned):
            if self.nuclear_slots[slot]:
                continue
            blend = self.chip_blend if self.chip_slots[slot] else self.fuel_blend
            flows[slot] = (1 - blend) * flows[slot] + blend * qty
        return {"flows": np.maximum(0.0, flows) * obs["action_mask"]}

    def _chain_plan(self, obs, baseline):
        if not np.all(obs["stock.qty.observed"] == 1):
            return None
        week = int(obs["week"][0])
        H = min(self.chain_horizon, self.T - week + 1)
        S, R = len(self.pairs), len(self.slot_edge)
        stock = np.asarray(obs["stock.qty"], dtype=float)
        cap = np.maximum(0.0, self._read(obs, "graph_now.u", self.nominal_cap))
        tau = np.maximum(1, self._read(obs, "graph_now.tau", self.nominal_tau)).astype(int)
        freight = self._read(obs, "graph_now.c", self.nominal_cost)
        tariff = self._read(obs, "graph_now.tariff", 0.0)
        supply = self._read(obs, "graph_now.supply.avail", 0.0)
        gbar = self._read(obs, "graph_now.grid.G_bar", self.grid_nominal)
        ybar = self._read(obs, "graph_now.grid.y_bar", [g["base_load"] for g in self.grid_attrs])
        restoration = self._read(obs, "graph_now.fab.R", 1.0)
        fabcap = self._read(obs, "graph_now.fab.cap_eff", [f["cap0"] for f in self.fab_attrs])
        oscap = self._read(obs, "graph_now.osat.thr_eff", [o["thr"] for o in self.osat_attrs])
        incoming = np.zeros((H, S))
        for t, node, k, qty in getattr(self, "_chain_arrivals", []):
            s = self.stock_index.get((node, k))
            if s is not None and 0 <= t < H:
                incoming[t, s] += qty
        wip_keys = ["wip." + key for key in ("node", "k", "qty", "out_week")]
        visible = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in wip_keys])
        for j in np.flatnonzero(visible):
            node, k, qty, out = (obs[key][j] for key in wip_keys)
            s = self.stock_index.get((int(node), int(k)))
            if s is not None and 0 <= int(out) - week < H:
                incoming[int(out) - week, s] += qty

        reopening, pending = {}, {}
        for prefix, labels, target in (
            ("closure_end.", ("chokepoint", "end_week"), reopening),
            ("pending_prohibitions.", ("edge", "k", "effective_week"), pending),
        ):
            keys = [prefix + label for label in labels]
            if not all(key in obs for key in keys):
                continue
            seen = np.logical_and.reduce([obs[key + ".observed"] == 1 for key in keys])
            for j in np.flatnonzero(seen):
                values = [int(obs[key][j]) for key in keys]
                if prefix == "closure_end.":
                    target[values[0]] = max(target.get(values[0], 0), values[1] + 1 - week)
                else:
                    pair = tuple(values[:2])
                    target[pair] = min(target.get(pair, self.T + 1), values[2] - week)
        capacities, _, edge_queue, _, _ = self._routing_state(baseline, obs, cap)

        def resource_cap(resource, t):
            node = self.edges["tail"][resource] if isinstance(resource, int) else resource[0]
            if node in reopening and t >= reopening[node]:
                return self.nominal_cap[resource] if isinstance(resource, int) else self.nominal_kappa[resource]
            return capacities[resource]

        # A dispatch consumes downstream capacity in its traversal week.
        # FIFO delays and known reopenings affect both arrival and that schedule.
        path_info = []
        for slot, path in enumerate(self.slot_paths):
            k, lead, resources = int(self.slot_k[slot]), 0, []
            for edge in path:
                node = self.edges["tail"][edge]
                if node in self.chk_pos:
                    pool = (node, self.pool[k])
                    wait = edge_queue.get(edge, 0.0) / max(resource_cap(edge, lead), 1e-6)
                    wait = min(H + 1, wait)
                    if node in reopening:
                        wait = min(wait, max(0, reopening[node] - lead))
                    lead += int(np.ceil(wait))
                    resources.append((lead, pool))
                resources.append((lead, int(edge)))
                lead += int(tau[edge])
            path_info.append((lead, resources))

        # Stock balance rows are the first H*S equalities.
        rhs = incoming.ravel().tolist()
        for s in range(S):
            rhs[s] += stock[s]
        eqr, eqc, eqv, ubr, ubc, ubv, limits = [], [], [], [], [], [], []
        costs, bounds = [], []

        def var(cost=0.0, upper=None, lower=0.0):
            col = len(costs)
            costs.append(float(cost) / 1e6)
            bounds.append((float(lower), None if upper is None else max(float(lower), float(upper))))
            return col

        def term(row, col, value):
            eqr.append(row)
            eqc.append(col)
            eqv.append(value)

        def equality(terms, value):
            row = len(rhs)
            rhs.append(float(value))
            for col, coefficient in terms:
                term(row, col, coefficient)

        def inequality(terms, value):
            row = len(limits)
            limits.append(float(value))
            for col, coefficient in terms:
                ubr.append(row)
                ubc.append(col)
                ubv.append(coefficient)

        discount = float(PARAMS.get("chain_discount", 0.97))
        weight = np.power(discount, np.arange(H))
        pi = np.asarray(self.static["sinks"]["pi"], dtype=float)
        backlog_on = self.static["sinks"]["backlog"]
        # End-of-window value follows the product to its market. At T use the
        # simulator's actual salvage instead. No reward at supply nodes.
        value_by_k = {
            k: max(pi[j] for j, (_, kk) in enumerate(self.layout["demands"]) if kk == k)
            for _, k in self.layout["demands"]
        }
        for osat in self.osat_attrs:
            for raw, product in osat["packages"].items():
                value_by_k[self.goods.index(raw)] = value_by_k.get(self.goods.index(product), 0.0)
        terminal_value = np.zeros(S)
        for s, (node, k) in enumerate(self.pairs):
            kind = self.static["nodes"]["type"][node]
            attrs = self.raw_nodes[node]["stock"][self.goods[k]]
            if week + H - 1 == self.T:
                terminal_value[s] = attrs.get("salvage", 0.0) if kind not in ("source", "material") else 0.0
            elif kind == "fab" and self.goods[k] == self.raw_nodes[node]["fab"]["input"]:
                terminal_value[s] = 0.15 * value_by_k.get(self.goods.index(self.raw_nodes[node]["fab"]["product"]), 0.0)
            elif kind in ("fab", "osat", "sink"):
                terminal_value[s] = 0.15 * value_by_k.get(k, 0.0)
            elif kind in ("terminal", "grid"):
                terminal_value[s] = 0.1 * max(g["voll"] for g in self.grid_attrs)
        inventory, X = np.zeros((H, S), dtype=int), np.zeros((H, R), dtype=int)
        outgoing, resource_terms = {}, {}
        for t in range(H):
            for s, (node, k) in enumerate(self.pairs):
                inventory[t, s] = var(
                    weight[t] * self._holding[s] - (weight[t] * terminal_value[s] if t == H - 1 else 0.0),
                    self._stock_limits[s],
                )
                term(t * S + s, inventory[t, s], 1.0)
                if t:
                    term(t * S + s, inventory[t - 1, s], -1.0)
                disposal = var(weight[t] * self.static["instance"]["commodities"][k]["disposal_cost"])
                term(t * S + s, disposal, 1.0)
            for j, pair in enumerate(self._supply_pairs):
                lift = var(upper=max(0.0, supply[j]))
                term(t * S + self.stock_index[pair], lift, -1.0)
            for slot, path in enumerate(self.slot_paths):
                k = int(self.slot_k[slot])
                source = self.stock_index[self.edges["tail"][path[0]], k]
                dest = self.stock_index[self.edges["head"][path[-1]], k]
                delay, resources = path_info[slot]
                upper = min(self.cap[slot], cap[path[0]]) if obs["action_mask"][slot] else 0.0
                if any(
                    t + departure >= pending.get((resource, k), self.T + 1)
                    for departure, resource in resources
                    if isinstance(resource, int)
                ):
                    upper = 0.0
                cost = weight[t] * sum(freight[e] + tariff[e, k] * self.value[k] for e in path)
                if t + delay >= H:
                    cost -= weight[t] * terminal_value[dest]
                col = X[t, slot] = var(cost, upper)
                term(t * S + source, col, 1.0)
                if t + delay < H:
                    term((t + delay) * S + dest, col, -1.0)
                outgoing.setdefault((t, source), []).append((col, 1.0))
                for departure, resource in resources:
                    if t + departure < H:
                        resource_terms.setdefault((t + departure, resource), []).append((col, 1.0))
            for s in range(S):
                terms = list(outgoing.get((t, s), []))
                if terms:
                    if t:
                        terms.append((inventory[t - 1, s], -1.0))
                    inequality(terms, stock[s] if t == 0 else 0.0)
        for (t, resource), terms in resource_terms.items():
            used = getattr(self, "_chain_existing_use", {}).get((t, resource), 0.0)
            inequality(terms, max(0.0, resource_cap(resource, t) - used))
        for t in range(H):
            for pool in set(self.pool):
                terms = []
                for slot, path in enumerate(self.slot_paths):
                    if self.pool[int(self.slot_k[slot])] != pool:
                        continue
                    lane = self.slot_lane[slot]
                    dt = sum(d for ln, d in self.fleet_terms.get(path[0], []) if ln is None or ln == lane)
                    if dt > 0:
                        terms.append((X[t, slot], dt))
                if terms:
                    inequality(terms, self.fleet_cap[pool])

        P = np.zeros((H, len(self.fab_attrs)), dtype=int)
        phat = []
        for fi, (node, fab) in enumerate(zip(self.layout["fabs"], self.fab_attrs)):
            si = self.stock_index[node, self.goods.index(fab["input"])]
            so = self.stock_index[node, self.goods.index(fab["product"])]
            phat.append(min(fabcap[fi], stock[si] + incoming[0, si]))
            for t in range(H):
                reward = terminal_value[so] * weight[t] if t + fab["tau"] >= H else 0.0
                col = P[t, fi] = var(-reward, fabcap[fi])
                term(t * S + si, col, 1.0)
                if t + fab["tau"] < H:
                    term((t + fab["tau"]) * S + so, col, -1.0)
        for oi, (node, osat) in enumerate(zip(self.layout["osats"], self.osat_attrs)):
            pairs = [(self.goods.index(r), self.goods.index(p)) for r, p in osat["packages"].items()]
            raw = [stock[self.stock_index[node, k]] + incoming[0, self.stock_index[node, k]] for k, _ in pairs]
            scale = min(1.0, oscap[oi] / max(sum(raw), 1e-9))
            for t in range(H):
                terms = []
                for j, (kr, kp) in enumerate(pairs):
                    si, so = self.stock_index[node, kr], self.stock_index[node, kp]
                    q = raw[j] * scale
                    reward = terminal_value[so] * weight[t] if t + osat["tau"] >= H else 0.0
                    col = var(-reward, q if t == 0 else oscap[oi], q if t == 0 else 0.0)
                    terms.append((col, 1.0))
                    term(t * S + si, col, 1.0)
                    if t + osat["tau"] < H:
                        term((t + osat["tau"]) * S + so, col, -1.0)
                inequality(terms, oscap[oi])

        psi = self.static["instance"]["params"]["psi"]
        for gi, (node, grid) in enumerate(zip(self.layout["grids"], self.grid_attrs)):
            members = self.grid_members[gi]
            for t in range(H):
                load = var(upper=1.0)
                energy_terms = []
                for name, share in grid["shares"].items():
                    segment = var(upper=share * gbar[gi])
                    energy_terms.append((segment, -1.0))
                    if name not in self.goods:
                        equality([(segment, 1.0), (load, -share * gbar[gi])], 0.0)
                        continue
                    k = self.goods.index(name)
                    s = self.stock_index[node, k]
                    term(t * S + s, segment, 1.0)
                    inequality([(segment, 1.0), (load, -share * gbar[gi])], 0.0)
                    short = var(weight[t] * self.value[k])
                    inequality([(load, share * gbar[gi]), (segment, -1.0), (short, -1.0)], 0.0)
                    if name == grid.get("rationed"):
                        threshold = psi * grid["ibar"][name]
                        terms = [(segment, threshold)]
                        if t:
                            terms.append((inventory[t - 1, s], -share * gbar[gi]))
                        inequality(terms, share * gbar[gi] * stock[s] if t == 0 else 0.0)
                served = var(upper=ybar[gi])
                # All Small grids serve base load before allocating fab energy.
                # Respect that automatic priority throughout this approximation.
                priority = 30e6 if grid["priority"] == "base_first" and members else 0.0
                shed = var(weight[t] * (grid["voll"] + priority), ybar[gi])
                equality([(served, 1.0), (shed, 1.0)], ybar[gi])
                energy_terms.append((served, 1.0))
                for fi in members:
                    e = self.fab_attrs[fi]["e"] / max(restoration[fi], 1e-9)
                    energy_terms.append((P[t, fi], e))
                inequality(energy_terms, 0.0)
                if t == 0:
                    ratio = var(upper=1.0)
                    for fi in members:
                        equality([(P[t, fi], 1.0), (ratio, -phat[fi])], 0.0)
        for fi, fab in enumerate(self.fab_attrs):
            if fab.get("grid") is None or fab["e"] == 0:
                equality([(P[0, fi], 1.0)], phat[fi])

        D = len(self.layout["demands"])
        forecast = self._read(obs, "demand_forecast.qty", 0.0)
        for di, (node, k) in enumerate(self.layout["demands"]):
            s = self.stock_index[node, k]
            nominal = self.raw_nodes[node]["sink"]["demand"][self.goods[k]]["dbar"]
            previous_backlog = None
            for t in range(H):
                demand = (
                    forecast[di, t] if t < forecast.shape[1] and obs["demand_forecast.qty.observed"][di, t] else nominal
                )
                served = var(upper=None if backlog_on[di] else demand)
                unserved = var(weight[t] * pi[di])
                term(t * S + s, served, 1.0)
                terms = [(served, 1.0), (unserved, 1.0)]
                initial_backlog = obs["backlog.qty"][di] if t == 0 and backlog_on[di] else 0.0
                if previous_backlog is not None:
                    terms.append((previous_backlog, -1.0))
                equality(terms, demand + initial_backlog)
                if backlog_on[di]:
                    cap_terms = [(served, 1.0)]
                    if previous_backlog is not None:
                        cap_terms.append((previous_backlog, -1.0))
                    inequality(cap_terms, demand + initial_backlog)
                    previous_backlog = unserved
        Aeq = coo_matrix((eqv, (eqr, eqc)), shape=(len(rhs), len(costs))).tocsr()
        Aub = coo_matrix((ubv, (ubr, ubc)), shape=(len(limits), len(costs))).tocsr()
        result = linprog(
            costs, A_eq=Aeq, b_eq=rhs, A_ub=Aub, b_ub=limits, bounds=bounds, method="highs", options={"time_limit": 0.6}
        )
        self.last_chain_diagnostics = {
            "status": int(result.status),
            "variables": len(costs),
            "equalities": len(rhs),
            "inequalities": len(limits),
            "demands": D,
        }
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        self.chain_solves += 1
        return np.minimum(self.cap, np.maximum(0.0, result.x[X[0]]))
