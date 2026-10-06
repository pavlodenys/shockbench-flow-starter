"""Score several recipes on the same episodes, and optionally the contribution of each atom (leave-one-out)."""

import copy
import json
import tempfile
from pathlib import Path

from modular.build import build, load_recipe
from modular.core import ATOMS, validate_recipe
from sbf_starter import scoring


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:+.{digits}f}"


def _pair(interval):
    return [float(interval[0]), float(interval[1])] if interval else None


def evaluate(
    recipes, task="small", episodes="dev", entropy=0, quick=False, cpu_budget=False, ablate=False, n_jobs=-1, log=print
):
    """Score recipes (dicts) on the same episodes; return the numbers as plain JSON values.

    ``{"scores": [{"name", "score", "interval", "cost_usd", "fallback_weeks"}], "ablation": [{"recipe", "atom",
    "contribution", "interval"}], "naive_cost_usd", "clairvoyant_cost_usd"}``. ``log(line)`` reports progress.
    """
    loaded = [validate_recipe(r) for r in recipes]
    names = [r["name"] for r in loaded]
    if len(set(names)) != len(names):
        raise ValueError(f"recipe names must differ, got {names}")
    log("reference costs (the first run on a network takes minutes, then they are cached)")
    es = scoring.episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=False)
    out = {"scores": [], "ablation": [], "task": task, "episodes": str(episodes), "entropy": entropy, "quick": quick}
    with tempfile.TemporaryDirectory(prefix="sbf-modular-") as tmp:
        work = Path(tmp)

        def folder(recipe, tag):
            path = work / f"{recipe['name']}__{tag}"
            path.mkdir()
            (path / "r.json").write_text(json.dumps(recipe))
            return build(path / "r.json", path)

        folders = {}
        for r in loaded:
            log(f"playing {r['name']}")
            folders[r["name"]] = folder(r, "full")
            s = es.score(folders[r["name"]], name=r["name"], cpu_budget=cpu_budget)
            out["scores"].append(
                {
                    "name": r["name"],
                    "score": s.rss,
                    "interval": _pair(s.interval),
                    "cost_usd": s.cost_usd,
                    "fallback_weeks": s.fallback_weeks,
                }
            )
            out["naive_cost_usd"], out["clairvoyant_cost_usd"] = s.naive_cost_usd, s.clairvoyant_cost_usd
        targets = names if ablate is True else list(ablate or [])  # True: every recipe; a list: those names
        if targets:
            for r in (r for r in loaded if r["name"] in targets):
                for i, entry in enumerate(r["atoms"]):
                    log(f"leave-one-out: {r['name']} without {entry['atom']}")
                    smaller = copy.deepcopy(r)
                    del smaller["atoms"][i]
                    smaller["name"] = f"{r['name']}-{entry['atom']}"
                    cmp = es.compare(folders[r["name"]], folder(smaller, "ablate"), cpu_budget=cpu_budget)
                    out["ablation"].append(
                        {
                            "recipe": r["name"],
                            "atom": entry["atom"],
                            "contribution": cmp.diff,
                            "interval": _pair(cmp.interval),
                        }
                    )
    return out


def run(recipes, task="small", episodes="dev", entropy=0, quick=False, cpu_budget=False, ablate=False, n_jobs=-1):
    """``evaluate`` on recipe files, printing the score table (and the ablation table); returns the same dict."""
    out = evaluate(
        [load_recipe(p) for p in recipes],
        task,
        episodes,
        entropy,
        quick,
        cpu_budget,
        ablate,
        n_jobs,
        log=lambda _: None,
    )
    print(
        f"\n{task}, episodes {episodes!r}, entropy {entropy}"
        + (", QUICK: not the leaderboard numbers" if quick else "")
    )
    print(f"{'recipe':<20}{'score':>9}   {'90% interval':<20}{'mean cost, USD':>20}{'naive weeks':>13}")
    for row in out["scores"]:
        lo, hi = row["interval"] or (None, None)
        interval = f"[{_fmt(lo, 3)}, {_fmt(hi, 3)}]"
        cost, weeks = row["cost_usd"], row["fallback_weeks"]
        print(f"{row['name']:<20}{_fmt(row['score']):>9}   {interval:<20}{cost:>20,.0f}{weeks:>13}")
    if ablate:
        print("\nleave-one-out: score of the recipe minus the score without the atom (paired, same episodes)")
        print(f"{'recipe / atom':<34}{'contribution':>13}   {'90% paired interval':<22}")
        for row in out["ablation"]:
            lo, hi = row["interval"] or (None, None)
            label = f"{row['recipe']} / {row['atom']}"
            print(f"{label:<34}{_fmt(row['contribution']):>13}   [{_fmt(lo, 3)}, {_fmt(hi, 3)}]")
        print("a positive contribution: the atom helps; an interval holding 0: the episodes cannot tell")
    return out


def neighbours(recipe, limit=24):
    """Recipes one step away from ``recipe``: each number moved a quarter of its range either way, each atom removed,
    each absent atom added with its defaults. ``[(label, kind, atom, param, value, recipe)]``.
    """
    recipe = validate_recipe(recipe)
    out = []

    def variant(tag, change):
        new = copy.deepcopy(recipe)
        new["name"] = f"{recipe['name']}__{tag}"
        change(new)
        return new

    for i, entry in enumerate(recipe["atoms"]):
        for key, spec in ATOMS[entry["atom"]].params.items():
            now = entry["params"][key]
            for sign_, word in ((-1, "менше"), (1, "більше")):
                value = round(min(spec.hi, max(spec.lo, now + sign_ * 0.25 * (spec.hi - spec.lo))), 4)
                if abs(value - now) < 1e-9:
                    continue
                new = variant(
                    f"{entry['atom']}_{key}_{len(out)}",
                    lambda r, i=i, k=key, v=value: r["atoms"][i]["params"].update({k: v}),
                )
                out.append(
                    (f"{entry['atom']}.{key}: {now:g} → {value:g} ({word})", "param", entry["atom"], key, value, new)
                )
        new = variant(f"drop_{entry['atom']}", lambda r, i=i: r["atoms"].pop(i))
        out.append((f"прибрати {entry['atom']}", "remove", entry["atom"], None, None, new))
    have = {e["atom"] for e in recipe["atoms"]}
    for name, cls in ATOMS.items():
        if name not in have:
            defaults = {k: p.default for k, p in cls.params.items()}
            new = variant(f"add_{name}", lambda r, n=name, d=defaults: r["atoms"].append({"atom": n, "params": d}))
            out.append((f"додати {name}", "add", name, None, None, validate_recipe(new)))
    return out[:limit]


def suggest(recipe, task="tiny", episodes="dev", entropy=0, quick=False, n_jobs=-1, log=print):
    """Score every neighbour of ``recipe`` against it on the same episodes (paired); best first.

    Each row: ``{"label", "kind", "atom", "param", "value", "diff", "interval", "recipe"}``; ``diff`` is the
    candidate's score minus the recipe's. Tune on your own root (``entropy`` != 0), confirm the winner on dev.
    """
    current = validate_recipe(recipe)
    es = scoring.episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sbf-modular-") as tmp:
        work = Path(tmp)

        def folder(r, tag):
            path = work / tag
            path.mkdir()
            (path / "r.json").write_text(json.dumps(r))
            return build(path / "r.json", path)

        base = folder(current, "current")
        candidates = neighbours(current)
        for n, (label, kind, atom, param, value, new) in enumerate(candidates, start=1):
            log(f"{n}/{len(candidates)}: {label}")
            cmp = es.compare(folder(new, f"c{n}"), base)
            rows.append(
                {"label": label, "kind": kind, "atom": atom, "param": param, "value": value, "diff": cmp.diff,
                 "interval": _pair(cmp.interval), "recipe": {**new, "name": current["name"]}}
            )  # fmt: skip
    rows.sort(key=lambda r: -(r["diff"] if r["diff"] is not None else -1e9))
    return {"suggestions": rows, "task": task, "episodes": str(episodes), "entropy": entropy, "quick": quick}
