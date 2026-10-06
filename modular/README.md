# modular: an agent built from small rules (atoms)

A **recipe** (JSON) lists atoms and their numbers; `build` turns it into a submission folder with one
self-contained `agent.py` (numpy and the standard library only, nothing imported from this repository). `compare` scores
recipes on the same episodes and, with `--ablate`, says what each atom contributes.

```bash
uv run python -m modular atoms                                           # the atoms, their stages and parameters
uv run python -m modular build modular/recipes/closure_cap.json --out agents/closure_cap
uv run sbf check closure_cap --task=small                                # the server's checks and a timed run
uv run python -m modular compare modular/recipes/*.json --task=small --ablate
uv run python -m modular compare a.json b.json --entropy=12345 --episodes=32   # tune on your own root, confirm on dev
```

## Dashboard

```bash
uv run python -m modular ui            # opens http://127.0.0.1:8765/ (--port=..., --noopen)
```

A page to assemble a policy by clicking: pick the base, add atoms, set their numbers with sliders, watch the recipe and
its validation update, then **build** `agents/<name>/agent.py`, **save** the recipe to `modular/recipes/`, or **compare**
it with the example recipes on Tiny or Small (`quick` or the 20 `dev` episodes, optionally with each atom's
leave-one-out contribution). Runs go in the background, one at a time. The server is standard library only, listens on
127.0.0.1, checks the `Host` header, writes only `agents/<name>` with names matching `[a-z][a-z0-9_]{0,39}`, and has no
upload endpoint. `uv run sbf check <name> --task=small` stays a command-line step.

The page also explains itself. Every atom card shows what its numbers do (live: "strait 50 % open → flow x0.50"),
which way each slider moves the behaviour, and what we measured. After a comparison, block 5 turns the numbers into
advice (does the recipe beat plain `empty`, which atoms help or hurt and what to try with each), and **Підібрати
покращення** runs `modular.compare.suggest`: every neighbour of the recipe (each number a quarter of its range up and
down, each atom removed, each absent atom added) is scored against the recipe on the same episodes, with a paired
interval, best first. By default it uses a root of its own (entropy 12345), not dev, so that the search does not fit the
dev episodes; confirm a winner on dev with Compare.

Nothing here uploads. `build` refuses to overwrite an `agent.py` it did not generate (`--force` overrides).

## What an agent does each week

1. **Base plan**: `"base": "capacity"` ships every allowed slot's nominal capacity (what `agents/template` does);
   `"nominal"` ships the week-0 shipments of the instance's initial pipeline. Both are multiplied by `action_mask`.
   Note: `nominal` is _not_ the naive rule that scores 0. The package's naive rule is a stateless order-up-to policy;
   repeating the nominal plan scored -2.39 on Small dev. Use `capacity` unless you want to experiment.
2. **Atoms**, in stage order: **`allow`** (what may be sent) → **`scale`** (how much) → **`stock`** (limits and
   targets from stock, demand and announcements). Within a stage, the recipe's order is kept. The order is fixed
   because a later stage must see what an earlier one decided: a demand cap should cap the already-reduced flow, and a
   front-load should raise from it. An atom's list position never overrides its stage.
3. **Guard** (always): an atom that raises, returns NaN, a wrong shape or a negative number is skipped (once noted on
   stderr); every flow is clipped into `[0, capacity * action_mask]`; if the final plan is still invalid the agent
   sends the maximum on allowed routes. `override_qty` and `release_mode` stay zero (the default release).

## Recipe format

```json
{
  "name": "closure_cap",
  "description": "optional",
  "base": "capacity",
  "atoms": [
    {"atom": "strait_open", "params": {"power": 1.0}},
    {"atom": "demand_cap", "params": {"cover": 4.0, "slack": 1.5}}
  ]
}
```

Unknown atoms, unknown parameters, non-numbers, values outside the declared `[lo, hi]` and a repeated atom raise a
`RecipeError` that names the atom and the valid choices. Omitted parameters take their defaults.

## The atoms

| atom                 | stage | what it does                                                                      | verdict on Small dev        |
| -------------------- | ----- | --------------------------------------------------------------------------------- | --------------------------- |
| `strait_open`        | scale | flow x (strait's open fraction) ** power, for lanes through a strait               | slightly hurts (-0.005)     |
| `fraction`           | scale | a fixed share of the base flow                                                    | hurts (0.8: -0.24)          |
| `warning_cut`        | scale | withhold a share through a strait whose `warning.score` exceeds a threshold       | ~0 (+0.0006)                |
| `demand_cap`         | stock | cap the total towards a market at what its forecast, backlog and stock still need | helps (+0.03, interval > 0) |
| `sanction_frontload` | stock | before an announced prohibition takes effect, raise affected slots to capacity    | ~0 (+0.0006)                |

Verdicts: leave-one-out on Small, dev (20 episodes) and own root 12345 (32 episodes), the same signs on both.
They are measurements of one setting, not proofs.

## Adding an atom

Create `modular/atoms/<name>.py`:

```python
import numpy as np

from modular.core import Atom, Param, register  # build:strip


@register
class MyAtom(Atom):
    """One line saying what it does (shown by `modular atoms`), then the details and the fields it reads."""

    name = "my_atom"
    stage = "scale"                                   # allow | scale | stock
    params = {"gain": Param(1.0, 0.0, 2.0, "what it means")}   # default, lo, hi

    def __init__(self, ctx, **values):                # optional: precompute from the network (ctx), once
        super().__init__(ctx, **values)

    def apply(self, plan, observation):               # plan.flows: one number per action slot
        plan.flows = plan.flows * self.p["gain"]
        return plan
```

Then add it to `modular/atoms/__init__.py`. Rules, because the file is pasted into the generated agent:

- imports: `numpy`, the standard library, or `scipy`/`torch`, one import per line, at the top level. The
  `from modular.core import ...  # build:strip` line is cut from the generated file.
- read only `self.ctx` (slot tables, capacities, strait matrix, `rng` seeded from `policy_seed`), `self.p`, the
  observation and your own state set in `__init__`. Use `seen(obs, key, default)` to ignore hidden values and
  `action_mask(obs, n)` for the mask. Never hold a reference to the observation between weeks.
- keep `apply` light: the budget is 2 s of CPU per week on Small (4 s on Full), and `Agent(config)` counts toward week 1.
- add a test in `tests/test_modular.py` on an edited observation.

## Files

`server.py` and `ui.html` (the dashboard) · `core.py` (parameters, plan, context, registry, validation, the agent and its guard; also pasted into every generated
agent) · `atoms/` · `build.py` · `compare.py` · `__main__.py` (the CLI) · `recipes/`.

## Known limits

- Tiny, Small and Full are read from `config`; the atoms were measured on Small only.
- `demand_cap` ignores cargo queued at straits and on earlier legs of a route.
- One flow per slot: no per-slot parameters, no use of `override_qty`/`release_mode` (queued tanker cargo is not steered).
- No atom reroutes around sanctions or expensive routes yet (`edges.alt_of` is the field to start from).
