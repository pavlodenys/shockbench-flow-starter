"""The core of the atom constructor: parameters, the plan an atom edits, the context, the registry and the agent.

This file is copied whole into every generated ``agent.py`` (``modular/build.py``), so it may import only the standard
library and numpy. Lines marked ``build:strip`` exist only in the repository and are cut from the generated file.
"""

import sys
from dataclasses import dataclass

import numpy as np


STAGES = ("allow", "scale", "stock")  # the order atoms are applied in; the base plan comes before, the guard after
BASES = ("nominal", "capacity")


class RecipeError(ValueError):
    """A recipe that cannot be built: the message says which atom or parameter and what would be valid."""


@dataclass(frozen=True)
class Param:
    """One numeric parameter of an atom: its default and the closed interval it must stay in."""

    default: float
    lo: float
    hi: float
    doc: str = ""


@dataclass
class Plan:
    """What the atoms edit: the three parts of an action, one array each."""

    flows: np.ndarray
    override_qty: np.ndarray
    release_mode: np.ndarray

    def copy(self):
        return Plan(self.flows.copy(), self.override_qty.copy(), self.release_mode.copy())


class Context:
    """Everything an atom may read about the network, derived once from ``config`` (no hidden state, no observation)."""

    def __init__(self, config):
        static, layout = config["static"], config["layout"]
        slots, edges = static["action_slots"], static["edges"]
        self.n_slots = len(slots["edge"])
        self.edge = np.asarray(slots["edge"], dtype=int)  # slot -> edge
        self.k = np.asarray(slots["k"], dtype=int)  # slot -> commodity
        self.lane = list(slots["lane"])  # slot -> lane index or None
        self.tail = np.asarray([edges["tail"][e] for e in self.edge], dtype=int)  # slot -> node the edge leaves
        self.head = np.asarray([edges["head"][e] for e in self.edge], dtype=int)  # slot -> node the edge reaches
        u0 = np.array([np.nan if edges["u0"][e] is None else edges["u0"][e] for e in self.edge], dtype=float)
        self.capacity = np.where(np.isfinite(u0), u0, 0.0)  # nominal capacity per slot
        lanes = static["lanes"]
        self.dest = np.array(  # slot -> the node its cargo finally reaches (the lane's last edge's head)
            [
                edges["head"][lanes["edges"][ln][-1]] if ln is not None else self.head[i]
                for i, ln in enumerate(self.lane)
            ],
            dtype=int,
        )
        self.route_edges = [  # slot -> every edge of its route (a lane's edges, else the slot's own edge)
            set(lanes["edges"][ln]) if ln is not None else {int(self.edge[i])} for i, ln in enumerate(self.lane)
        ]
        self.override_shape = tuple(config["spaces"]["action"]["override_qty"]["shape"])
        self.release_shape = tuple(config["spaces"]["action"]["release_mode"]["shape"])
        self.rng = np.random.default_rng(config["policy_seed"])  # all randomness comes from here
        self.layout = layout
        self.static = static
        self.T = int(config["T"])
        # strait_matrix[s, c] = 1 where slot s's lane passes the c-th strait (row of graph_now.open)
        position = {node: i for i, node in enumerate(layout["chokepoints"])}
        self.strait_matrix = np.zeros((self.n_slots, len(position)))
        for s, lane in enumerate(self.lane):
            if lane is not None:
                for node in static["lanes"]["chokepoints"][lane]:
                    self.strait_matrix[s, position[node]] = 1.0
        # the normal plan: the week-0 shipments of the instance's initial pipeline, per (edge, commodity, lane)
        ids = {"edge": static["edges"]["id"], "k": static["commodities"]["id"], "lane": static["lanes"]["id"]}
        shipped = {}
        for p in static["instance"]["initial_state"]["pipeline"]:
            if p.get("dispatch_week", 0) == 0:
                key = (p["edge"], p["k"], p.get("lane"))
                shipped[key] = shipped.get(key, 0.0) + float(p["qty"])
        self.nominal = np.array(
            [
                shipped.get(
                    (
                        ids["edge"][self.edge[s]],
                        ids["k"][self.k[s]],
                        None if self.lane[s] is None else ids["lane"][self.lane[s]],
                    ),
                    0.0,
                )
                for s in range(self.n_slots)
            ]
        )


def seen(observation, key, default=0.0):
    """The array ``key`` where its ``.observed`` mask is 1, ``default`` elsewhere (padding and hidden values)."""
    return np.where(observation[key + ".observed"] == 1, observation[key], default)


def action_mask(observation, n):
    """1 where a slot may carry goods this week; every slot is allowed in a blackout week."""
    if observation["action_mask.observed"][0] == 1:
        return np.asarray(observation["action_mask"], dtype=float)
    return np.ones(n)


class Atom:
    """A small rule that edits a plan. Subclass, set ``name``/``stage``/``params`` and write ``apply``.

    ``apply(plan, observation)`` returns the edited plan (or the same one). It may read ``self.ctx`` and the
    observation and keep its own state in ``self`` (set in ``__init__``); nothing else. Stages: ``allow`` (what may
    be sent), ``scale`` (how much), ``stock`` (limits and targets from stock and demand).
    """

    name = ""
    stage = "scale"
    doc = ""
    ui = ""  # a one-sentence description in Ukrainian for the dashboard (falls back to the docstring)
    params = {}

    def __init__(self, ctx, **values):
        self.ctx = ctx
        self.p = {k: float(spec.default) for k, spec in self.params.items()} | {k: float(v) for k, v in values.items()}

    def apply(self, plan, observation):
        raise NotImplementedError


ATOMS = {}  # name -> class, filled by @register


def register(cls):
    if not cls.name or cls.stage not in STAGES:
        raise TypeError(f"{cls.__name__}: needs a name and a stage in {STAGES}")
    ATOMS[cls.name] = cls
    return cls


def validate_recipe(recipe):
    """Check a recipe (a dict) and return it normalised: defaults filled in, atoms sorted by stage (stable).

    A recipe is ``{"name": str, "base": "nominal"|"capacity", "atoms": [{"atom": str, "params": {...}}, ...]}``.
    """
    if not isinstance(recipe, dict):
        raise RecipeError("a recipe is a JSON object")
    extra = set(recipe) - {"name", "description", "base", "atoms"}
    if extra:
        raise RecipeError(f"unknown recipe key(s) {sorted(extra)}; valid: name, description, base, atoms")
    if recipe.get("base") not in BASES:
        raise RecipeError(f"base must be one of {list(BASES)}, got {recipe.get('base')!r}")
    out, used = [], set()
    for i, entry in enumerate(recipe.get("atoms", [])):
        name = entry.get("atom") if isinstance(entry, dict) else None
        if name not in ATOMS:
            raise RecipeError(f"atoms[{i}]: unknown atom {name!r}; known: {sorted(ATOMS)}")
        if name in used:
            raise RecipeError(f"atoms[{i}]: {name!r} appears twice; list each atom once")
        used.add(name)
        specs = ATOMS[name].params
        given = entry.get("params", {})
        bad = set(given) - set(specs)
        if bad:
            raise RecipeError(f"atom {name!r}: unknown parameter(s) {sorted(bad)}; valid: {sorted(specs)}")
        values = {}
        for key, spec in specs.items():
            value = given.get(key, spec.default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RecipeError(f"atom {name!r}: parameter {key!r} must be a finite number, got {value!r}")
            if not spec.lo <= value <= spec.hi:
                raise RecipeError(f"atom {name!r}: parameter {key!r} = {value} is outside [{spec.lo}, {spec.hi}]")
            values[key] = value
        out.append({"atom": name, "params": values})
    out.sort(key=lambda e: STAGES.index(ATOMS[e["atom"]].stage))
    name, description = recipe.get("name", "recipe"), recipe.get("description", "")
    return {"name": name, "description": description, "base": recipe["base"], "atoms": out}


RECIPE = {}  # the generated agent.py sets this to its validated recipe, after this file's code


class Agent:
    """The generated agent: base plan, then the atoms in stage order, then the guard. Falls back, never raises."""

    def __init__(self, config=None):
        self.ctx = Context(config)
        recipe = validate_recipe(RECIPE)
        self.base = recipe["base"]
        self.atoms = [ATOMS[e["atom"]](self.ctx, **e["params"]) for e in recipe["atoms"]]
        self.override_qty = np.zeros(self.ctx.override_shape)
        self.release_mode = np.zeros(self.ctx.release_shape, dtype=np.int64)  # 0: the default release
        self.warned = set()

    def _safe(self, mask):
        """The plan played when something is wrong: send the maximum on the routes that are allowed (template's)."""
        return self.ctx.capacity * mask

    def _fresh(self, observation):
        ctx = self.ctx
        mask = action_mask(observation, ctx.n_slots)
        flows = (ctx.capacity if self.base == "capacity" else ctx.nominal) * mask
        return Plan(flows.astype(float), self.override_qty.copy(), self.release_mode.copy()), mask

    @staticmethod
    def _valid(plan, ctx, mask):
        f = plan.flows
        return (
            f.shape == (ctx.n_slots,)
            and np.all(np.isfinite(f))
            and np.all(f >= 0.0)
            and np.all(f <= ctx.capacity * mask * (1 + 1e-9) + 1e-9)
            and plan.override_qty.shape == ctx.override_shape
            and np.all(np.isfinite(plan.override_qty))
            and plan.release_mode.shape == ctx.release_shape
            and np.all(np.isin(plan.release_mode, (0, 1, 2)))
        )

    def act(self, observation):
        ctx = self.ctx
        try:
            plan, mask = self._fresh(observation)
        except Exception as error:  # the observation is unreadable: nothing to build a plan on
            print(f"modular agent: no base plan ({error!r}); sending the maximum", file=sys.stderr)
            return {"flows": ctx.capacity.copy(), "override_qty": self.override_qty, "release_mode": self.release_mode}
        for atom in self.atoms:
            try:
                candidate = atom.apply(plan.copy(), observation)
                clipped = Plan(
                    np.clip(np.nan_to_num(candidate.flows, nan=0.0, posinf=0.0, neginf=0.0), 0.0, ctx.capacity * mask),
                    candidate.override_qty,
                    candidate.release_mode,
                )
                if self._valid(clipped, ctx, mask) and np.all(np.isfinite(candidate.flows)):
                    plan = clipped
                elif atom.name not in self.warned:
                    self.warned.add(atom.name)
                    print(f"modular agent: atom {atom.name!r} returned an invalid plan; skipped", file=sys.stderr)
            except Exception as error:
                if atom.name not in self.warned:
                    self.warned.add(atom.name)
                    print(f"modular agent: atom {atom.name!r} raised {error!r}; skipped", file=sys.stderr)
        if not self._valid(plan, ctx, mask):
            plan = Plan(self._safe(mask), self.override_qty.copy(), self.release_mode.copy())
        return {"flows": plan.flows, "override_qty": plan.override_qty, "release_mode": plan.release_mode}
