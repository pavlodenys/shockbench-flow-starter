"""An interactive explorer of dev episodes: one HTML file to click through the network week by week.

    uv run python examples/09_explorer.py
    uv run python examples/09_explorer.py --agents=template,heuristic,mine --episodes=29,5 --quick

Plays each agent on each episode, then writes explorer.html: the map of the network, what the agent observes
each week (named in plain words, with the field's code beside it), what it does, and what it costs. Open the file in a
browser; it needs no server. The labels are Ukrainian and written for Tiny; on Small and Full the nodes and edges show
their codes. The naive rule takes about 30 s on Tiny the first time without --quick, then comes from the cache.
"""

import json
import re
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from shockbench_flow_agent import NAIVE_REPLICATIONS, QUICK
from shockbench_flow_gym import dashboard  # also registers the ShockBench/* environments

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


HERE = Path(__file__).resolve().parent

AGENT_LABELS = {
    "template": "Слати максимум",
    "heuristic": "Евристика (протока)",
    "random": "Випадковий",
}

# node id -> (short label for the map, full label)
NODES = {
    "src_gulf": ("Затока", "Перська затока: газ (LNG)"),
    "src_usau": ("США/Австралія", "США та Австралія: газ (LNG)"),
    "src_ru": ("Росія", "Росія: газ (LNG)"),
    "chk": ("Протока", "Морська протока (вузьке місце)"),
    "grid_tw": ("Мережа Тайваню", "Енергомережа Тайваню"),
    "grid_eu": ("Мережа ЄС", "Енергомережа ЄС"),
    "mat_jp": ("Матеріали, Японія", "Японія: кремнієві пластини"),
    "fab_tw": ("Фабрика Тайвань", "Фабрика чипів, Тайвань"),
    "fab_cn": ("Фабрика Китай", "Фабрика чипів, Китай"),
    "fab_us": ("Фабрика США", "Фабрика чипів, США"),
    "osat_sea": ("Пакування (OSAT)", "Пакування й тестування чипів (OSAT), Південно-Східна Азія"),
    "sink_us": ("Ринок США", "Кінцевий попит, США"),
}
NODE_TYPES = {
    "source": "джерело сировини",
    "material": "постачальник матеріалів",
    "grid": "енергомережа",
    "chokepoint": "морська протока",
    "fab": "фабрика чипів",
    "osat": "пакування й тестування",
    "sink": "ринок (попит)",
    "terminal": "термінал",
}
COMMODITIES = {
    "lng": ("Газ (LNG)", "ГВт·год палива"),
    "wafer": ("Кремнієві пластини", "пластин-екв. 300 мм"),
    "chip_le_raw": ("Чипи неупаковані", "пластин-екв. 300 мм"),
    "chip_le": ("Чипи готові", "пластин-екв. 300 мм"),
}
MODES = {"sea": "море", "air": "повітря", "pipeline": "трубопровід", "grid": "енергозв'язок"}
REGIONS = {
    "TW": "Тайвань",
    "KR": "Півд. Корея",
    "JP": "Японія",
    "CN": "Китай",
    "US": "США",
    "EU": "ЄС",
    "GULF": "Перська затока",
    "RU": "Росія",
    "AU": "Австралія",
    "SEA": "Півд.-Сх. Азія",
    "IN": "Індія",
    "UA": "Україна",
    "KZ": "Казахстан",
    "ROW": "Решта світу",
}
COSTS = {
    "freight": ("Перевезення", "плата за доставку вантажу"),
    "war_risk": ("Воєнний ризик", "страхова надбавка за небезпечну протоку"),
    "tariff": ("Мита", "податок на імпорт за ребром і товаром"),
    "holding": ("Зберігання", "утримання запасів на складах"),
    "queue_holding": ("Черга біля протоки", "вантаж чекає на проходження"),
    "shortage": ("Дефіцит", "штраф за попит, якого не задоволено"),
    "disposal": ("Утилізація", "списання зайвого або непридатного"),
    "shed": ("Відключення енергії", "штраф за недоподану електроенергію"),
}
CHANNELS = [
    "офіційна пропозиція тарифу",
    "неофіційний тариф",
    "остаточний тариф",
    "санкції (юридичні)",
    "загроза розірвати зв'язки",
    "загроза навколо протоки",
]
KINDS = ["пропозиція", "остаточне повідомлення", "загроза", "публікація", "відкликання"]
WAR_RISK = ["немає", "Червоне море", "Ормуз 2026"]
STAGE = {"source": 0, "material": 0, "grid": 1, "fab": 2, "osat": 3, "sink": 4, "terminal": 4}
SEVERITY = {"chokepoint": "bad", "prohibited": "bad", "war_risk": "bad", "capacity": "warn", "grid": "warn"}


def clean(x, digits=3):
    """Nested lists with NaN as null and floats rounded, to keep the JSON small."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return [clean(v, digits) for v in x]
    if isinstance(x, float):
        return None if x != x else round(x, digits)
    return x


class Exporter:
    def __init__(self, static: dict, meta: dict, layout: dict):
        self.s, self.meta, self.layout = static, meta, layout
        self.nid = static["nodes"]["id"]
        self.ntype = static["nodes"]["type"]
        self.eid = static["edges"]["id"]
        self.kid = static["commodities"]["id"]
        self.regions = static["regions"]

    def node(self, i):
        return NODES.get(self.nid[i], (self.nid[i], self.nid[i]))[0]

    def edge(self, e):
        t, h = self.s["edges"]["tail"][e], self.s["edges"]["head"][e]
        return f"{self.node(t)} → {self.node(h)}"

    def commodity(self, k):
        return COMMODITIES.get(self.kid[k], (self.kid[k],))[0]

    def region(self, r):
        return REGIONS.get(self.regions[r], self.regions[r])

    def lane(self, lane):
        edges = self.s["lanes"]["edges"][lane]
        heads = [self.s["edges"]["head"][e] for e in edges]
        path = [self.s["edges"]["tail"][edges[0]], *heads]
        return " → ".join(self.node(n) for n in path)

    def network(self) -> dict:
        s = self.s
        columns: dict[int, list[int]] = {}
        for i, kind in enumerate(self.ntype):
            if kind != "chokepoint":
                columns.setdefault(STAGE.get(kind, 2), []).append(i)
        pos = {}
        for col, members in columns.items():
            for row, i in enumerate(members):
                pos[i] = (0.07 + col * 0.215, 0.24 + (row + 0.5) / len(members) * 0.72 - 0.0)
        chokepoints = [i for i, kind in enumerate(self.ntype) if kind == "chokepoint"]
        for n, i in enumerate(chokepoints):
            pos[i] = (0.5 + (n - (len(chokepoints) - 1) / 2) * 0.16, 0.07)
        nodes = [
            {
                "id": self.nid[i],
                "type": self.ntype[i],
                "typeLabel": NODE_TYPES.get(self.ntype[i], self.ntype[i]),
                "short": self.node(i),
                "label": NODES.get(self.nid[i], (None, self.nid[i]))[1],
                "region": self.region(s["nodes"]["region"][i]),
                "x": round(pos[i][0], 4),
                "y": round(pos[i][1], 4),
            }
            for i in range(len(self.nid))
        ]
        edges = []
        for e, code in enumerate(self.eid):
            ks = s["edges"]["K"][e]
            edges.append(
                {
                    "id": code,
                    "tail": s["edges"]["tail"][e],
                    "head": s["edges"]["head"][e],
                    "label": self.edge(e),
                    "mode": MODES.get(s["edges"]["mode"][e], s["edges"]["mode"][e]),
                    "k": ks,
                    "kLabel": ", ".join(self.commodity(k) for k in ks) or "енергія",
                    "tau0": s["edges"]["tau0"][e],
                    "c0": s["edges"]["c0"][e],
                    "u0": s["edges"]["u0"][e],
                    "duplicate": s["edges"]["alt_of"][e] is not None,
                }
            )
        slots = []
        sl = s["action_slots"]
        for i, (e, k, lane) in enumerate(zip(sl["edge"], sl["k"], sl["lane"])):
            on_lane = lane is not None
            slots.append(
                {
                    "edge": e,
                    "k": k,
                    "lane": lane,
                    "label": f"{self.edge(e)}: {self.commodity(k)}",
                    "route": self.lane(lane) if on_lane else None,
                    "laneEdges": s["lanes"]["edges"][lane] if on_lane else [e],
                    "u0": s["edges"]["u0"][e],
                }
            )
        override = [
            {
                "chokepoint": c,
                "k": k,
                "outEdge": o,
                "lane": lane,
                "label": f"{self.commodity(k)} через «{self.node(c)}» далі {self.edge(o)}",
            }
            for c, k, o, lane in zip(*(s["override_slots"][f] for f in ("chokepoint", "k", "out_edge", "lane")))
        ]
        units = []
        for kind, i in self.layout["warning_units"]:
            if kind == "region":
                units.append(f"регіон: {self.region(i)}")
            elif kind == "dyad":
                units.append(f"пара суперників: {self.region(s['dyads']['a'][i])} і {self.region(s['dyads']['b'][i])}")
            else:
                units.append(f"протока: {self.node(i)}")
        sinks = s["sinks"]
        return {
            "instance": s["instance_id"],
            "T": s["T"],
            "nodes": nodes,
            "edges": edges,
            "slots": slots,
            "override": override,
            "commodities": [
                {"id": c, "label": COMMODITIES.get(c, (c, ""))[0], "unit": COMMODITIES.get(c, (c, ""))[1]}
                for c in self.kid
            ],
            "stockSlots": [{"node": n, "k": k} for n, k in self.layout["stock_slots"]],
            "supplySlots": [{"node": n, "k": k} for n, k in self.layout["supply_slots"]],
            "releasePairs": [{"node": n, "k": k} for n, k in self.layout["release_pairs"]],
            "warningUnits": units,
            "chokepoints": self.layout["chokepoints"],
            "fabs": self.layout["fabs"],
            "grids": self.layout["grids"],
            "osats": self.layout["osats"],
            "sinks": [
                {"node": n, "k": k, "pi": p, "backlog": b}
                for n, k, p, b in zip(sinks["node"], sinks["k"], sinks["pi"], sinks["backlog"])
            ],
            "costs": [
                {"id": c, "label": COSTS.get(c, (c, ""))[0], "hint": COSTS.get(c, ("", ""))[1]}
                for c in self.meta["cost_components"]
            ],
        }

    def message(self, ch, kind, region, tk, target, k, ann, eff):
        what = ["протоку", "ребро", "вузол", "регіон"][tk]
        if tk == 0:
            name = self.node(target)
        elif tk == 1:
            name = self.edge(target)
        elif tk == 2:
            name = self.node(target)
        else:
            name = self.region(target)
        text = f"{CHANNELS[ch].capitalize()} ({KINDS[kind]}). Ціль, {what}: {name}"
        if k is not None:
            text += f"; товар: {self.commodity(k)}"
        if eff is not None and eff > 0:
            text += f"; діє з тижня {eff}"
        return text

    def event(self, e):
        sig, subj, before, after = e["signal"], e["subject"], e["before"], e["after"]
        edge_k = re.fullmatch(r"(E\d+) (\w+)", subj)
        if edge_k:
            ei, k = self.eid.index(edge_k[1]), self.kid.index(edge_k[2])
            where = f"{self.edge(ei)} ({self.commodity(k)})"
        elif re.fullmatch(r"E\d+", subj):
            where = self.edge(self.eid.index(subj))
        elif subj in self.nid:
            where = self.node(self.nid.index(subj))
        else:
            where = subj
        if sig == "prohibited":
            title = f"Заборона набула чинності: {where}" if before in (0, None) and e["note"] != "in force" else f"Діє заборона: {where}"
        elif sig == "lifted":
            title = f"Заборону знято: {where}"
        elif sig == "capacity":
            title = f"Потужність {where}: {before:g} → {after:g}"
        elif sig == "tariff":
            before_text = "" if before is None else f"{before:.0%} → "
            title = f"Мито {where}: {before_text}{after:.0%}"
        elif sig == "grid":
            title = f"Генерація в мережі «{where}»: {before:.1f} → {after:.1f}"
        elif sig == "chokepoint":
            title = f"Протока «{where}» відкрита на {after:.1%} (було {before:.1%})"
        elif sig == "war_risk":
            old = {"none": "немає", "red_sea": "Червоне море", "hormuz_2026": "Ормуз 2026"}
            title = f"Воєнний ризик у протоці: {old.get(before, before)} → {old.get(after, after)}"
        elif sig.startswith("message"):
            m = re.match(r"(\w+) (\w+) on (\w+) (.+?)(?: \((\w+)\))?$", subj)
            title = {"message": "Нове повідомлення", "message_update": "Оновлення повідомлення", "message_ended": "Повідомлення закрито"}[sig]
            if m:
                ch = ["tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat"].index(m[1])
                target = m[4]
                if m[3] == "edge":
                    target = self.edge(self.eid.index(target))
                elif m[3] == "region":
                    target = self.region(self.regions.index(target))
                elif target in self.nid:
                    target = self.node(self.nid.index(target))
                kind = f", {self.commodity(self.kid.index(m[5]))}" if m[5] else ""
                title += f": {CHANNELS[ch]} ({target}{kind})"
        else:
            title = f"{sig}: {where}"
        return {"week": e["week"], "kind": sig, "severity": SEVERITY.get(sig, "info"), "title": title}


def agent_data(rec: dict, ex: Exporter, net: dict) -> dict:
    """One agent's episode as columns over weeks: the observation, the action and the outcome."""
    obs, T = rec["obs"], rec["meta"]["T"]
    n_obs = len(next(iter(obs.values())))

    def col(key, shift=0):
        v = np.asarray(obs[key])[shift : shift + T]
        seen = obs.get(key + ".observed")
        if seen is not None and np.asarray(seen).shape == np.asarray(obs[key]).shape:
            seen = np.asarray(seen)[shift : shift + T] == 1
            v = np.where(seen, v, np.nan).astype(float)
        return v

    def rows(prefix, fields):
        """The padded lists of a prefix as one list of row-lists per week, live entries only."""
        out = []
        for w in range(T):
            live = np.asarray(obs[f"{prefix}.{fields[0]}.observed"])[w] == 1
            out.append([[obs[f"{prefix}.{f}"][w][i].item() for f in fields] for i in np.where(live)[0]])
        return out

    def observed_or_none(prefix, field, w, i):
        return obs[f"{prefix}.{field}"][w][i].item() if obs[f"{prefix}.{field}.observed"][w][i] == 1 else None

    d = {}
    d["stock"] = col("stock.qty")
    d["backlog"] = col("backlog.qty")
    d["open"] = col("graph_now.open")
    d["kappaTb"] = col("graph_now.kappa.tb")
    d["kappaCt"] = col("graph_now.kappa.ct")
    d["warRisk"] = col("graph_now.war_risk")
    d["u"] = col("graph_now.u")
    d["c"] = col("graph_now.c")
    d["tau"] = col("graph_now.tau")
    prohibited = np.asarray(obs["graph_now.prohibited"])[:T]
    tariff = np.asarray(obs["graph_now.tariff"])[:T]
    d["prohibited"] = [[[int(e), int(k)] for e, k in zip(*np.where(prohibited[w] == 1))] for w in range(T)]
    d["tariff"] = [[[int(e), int(k), float(tariff[w][e][k])] for e, k in zip(*np.where(tariff[w] > 0))] for w in range(T)]
    d["supply"] = col("graph_now.supply.avail")
    d["fabCap"] = col("graph_now.fab.cap_eff")
    d["fabR"] = col("graph_now.fab.R")
    d["gridG"] = col("graph_now.grid.G_bar")
    d["gridY"] = col("graph_now.grid.y_bar")
    d["osatThr"] = col("graph_now.osat.thr_eff")
    d["osatR"] = col("graph_now.osat.R")
    d["warning"] = col("warning.score")
    d["forecast"] = col("demand_forecast.qty").reshape(T, -1)
    d["mask"] = np.asarray(obs["action_mask"])[:T]
    d["overrideMask"] = np.asarray(obs["override_mask"])[:T]
    d["lastReq"] = col("last_week.clip.requested")
    d["lastExec"] = col("last_week.clip.executed")
    d["lastDemand"] = col("last_week.sinks.demand")
    d["lastServed"] = col("last_week.sinks.served")
    d["lastLost"] = col("last_week.sinks.lost")
    d["lastShed"] = col("last_week.shed.qty")
    d["lastCosts"] = col("last_week.cost_components")
    d["pipeline"] = rows("pipeline", ["edge", "k", "qty", "arrival_week"])
    d["queue"] = rows("queue_lots", ["lot_id", "chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week"])
    d["wip"] = rows("wip", ["node", "k", "qty", "out_week"])
    d["pending"] = rows("pending_prohibitions", ["edge", "k", "effective_week"])
    d["closureEnd"] = rows("closure_end", ["chokepoint", "end_week"])
    messages = []
    for w in range(T):
        live = np.asarray(obs["messages.msg_id.observed"])[w] == 1
        week = []
        for i in np.where(live)[0]:
            k = observed_or_none("messages", "k", w, i)
            eff = observed_or_none("messages", "stated_effective_week", w, i)
            ch, kind = obs["messages.channel"][w][i].item(), obs["messages.kind"][w][i].item()
            region, tk = obs["messages.region"][w][i].item(), obs["messages.target_kind"][w][i].item()
            target, ann = obs["messages.target"][w][i].item(), obs["messages.announced_week"][w][i].item()
            week.append(
                {
                    "id": obs["messages.msg_id"][w][i].item(),
                    "channel": ch,
                    "channelLabel": CHANNELS[ch],
                    "kind": kind,
                    "announced": ann,
                    "effective": eff,
                    "chokepointThreat": ch == 5,
                    "text": ex.message(ch, kind, region, tk, target, k, ann, eff),
                }
            )
        messages.append(week)
    d["messages"] = messages
    d["flows"] = np.asarray(rec["action"]["flows"])
    d["overrideQty"] = np.asarray(rec["action"]["override_qty"])
    d["releaseMode"] = np.asarray(rec["action"]["release_mode"])
    d["exec"] = col("last_week.clip.executed", 1)
    d["costs"] = np.asarray(rec["costs"])
    d["rewardCents"] = np.asarray(rec["reward_cents"])
    d["demand"] = col("last_week.sinks.demand", 1)
    d["served"] = col("last_week.sinks.served", 1)
    d["lost"] = col("last_week.sinks.lost", 1)
    d["shed"] = col("last_week.shed.qty", 1)
    assert n_obs == T + 1
    return {k: clean(v) for k, v in d.items()} | {"J": rec["meta"]["J_cents"] / 100, "salvage": rec["meta"]["salvage_cents"] / 100}


def main(
    task: str = "tiny",
    agents: str = "template,heuristic,random",
    episodes: str | None = None,
    count: int = 3,
    search: int = 60,
    min_open: float = 0.5,
    quick: bool = False,
    n_jobs: int = -1,
    regime: str = "standard",
    seed: int = 0,
    out: str | None = None,
) -> None:
    """Write explorer.html.

    Args:
        task: tiny, small or full.
        agents: comma-separated agent names (folders of agents/) to play.
        episodes: comma-separated dev episodes (default: the first ``count`` in which a strait closes).
        count: how many episodes to take when ``episodes`` is not given.
        search: look through dev episodes 0 .. search - 1.
        min_open: the open fraction a strait must fall below.
        quick: a rough naive rule, not the board's.
        n_jobs: workers of the naive rule's first computation (-1: all cores).
        regime: the information regime; standard is the scored one.
        seed: the reset's seed.
        out: the run folder (default: outputs/09_explorer/<date_time>).
    """
    out = Path(out or f"outputs/09_explorer/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id(task), regime=regime)
    names = [a.strip() for a in agents.split(",") if a.strip()]
    if episodes is None:
        chosen = episodes_with_closure(env, count, search=search, min_open=min_open, seed=seed)
    else:
        chosen = [int(n) for n in str(episodes).split(",")] if not isinstance(episodes, (tuple, list)) else [int(n) for n in episodes]
    print(f"episodes {chosen}; agents {names}")
    replications = QUICK["fq_replications"] if quick else NAIVE_REPLICATIONS
    exporter, network, data = None, None, []
    for n in chosen:
        entry = {"episode": n, "agents": {}, "events": [], "naive": None}
        for name in names:
            start = time.perf_counter()
            rec = dashboard.record_episode(
                env, load(name), seed=seed, options={"episode": n}, naive_replications=replications, n_jobs=n_jobs
            )
            if exporter is None:
                exporter = Exporter(rec["static"], rec["meta"], dashboard.layout_tables(env.unwrapped.layout))
                network = exporter.network()
            entry["agents"][name] = agent_data(rec, exporter, network) | {"label": AGENT_LABELS.get(name, name)}
            if entry["naive"] is None:
                entry["naive"] = {
                    "J": rec["meta"]["naive_J_cents"] / 100,
                    "costs": clean(np.asarray(rec["naive"]["costs"])),
                }
                entry["events"] = [exporter.event(e) for e in rec["events"]]
            print(f"  episode {n} {name}: cost ${rec['meta']['J_cents'] / 100:,.0f} ({time.perf_counter() - start:.1f} s)")
        data.append(entry)
    payload = {
        "task": task,
        "regime": regime,
        "quick": quick,
        "network": network,
        "agentOrder": names,
        "episodes": data,
        "warWarning": WAR_RISK,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (HERE / "09_explorer.html").read_text(encoding="utf-8").replace("__DATA__", blob)
    page = f'<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n{html}'
    (out / "explorer.html").write_text(page, encoding="utf-8")
    print(f"written in {out}: explorer.html ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    fire.Fire(main)
