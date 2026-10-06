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
                g = self.group_for.get((dest, k))
                if g is not None:
                    eta = t - week + 1 + (dest != self.groups[g][1])
                    if tag == -1:
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
