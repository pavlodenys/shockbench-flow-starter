"""The atom constructor: recipe validation, atoms on edited observations, the guard, the generated agent."""

import ast
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from shockbench_flow_agent import agent_config

import modular.atoms  # noqa: F401 - registers the atoms
from modular import build as modular_build
from modular.core import ATOMS, Atom, Context, Param, Plan, RecipeError, register, validate_recipe
from sbf_starter import agents
from sbf_starter import cli as sbf
from tests.conftest import ROOT


RECIPES = ROOT / "modular" / "recipes"


@pytest.fixture(scope="module")
def tiny():
    env = gym.make("ShockBench/Tiny-v0")
    obs, info = env.reset(options={"episode": 0})
    return env, obs, agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)


def recipe(base="capacity", **atoms):
    return {"name": "t", "base": base, "atoms": [{"atom": a, "params": p} for a, p in atoms.items()]}


def fresh_plan(ctx, flows=None):
    flows = ctx.capacity.copy() if flows is None else flows
    return Plan(flows, np.zeros(ctx.override_shape), np.zeros(ctx.release_shape, dtype=np.int64))


# --- recipes -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"name": "x", "base": "max", "atoms": []}, "base must be one of"),
        (recipe(nope={}), "unknown atom 'nope'"),
        (recipe(strait_open={"powr": 1}), "unknown parameter"),
        (recipe(strait_open={"power": 9}), "outside \\[0.0, 5.0\\]"),
        (recipe(strait_open={"power": "1"}), "must be a finite number"),
        (recipe(strait_open={"power": float("nan")}), "must be a finite number"),
        ({"base": "capacity", "atoms": [{"atom": "fraction"}, {"atom": "fraction"}]}, "appears twice"),
        ({"base": "capacity", "atomz": []}, "unknown recipe key"),
    ],
)
def test_invalid_recipes_name_the_problem(bad, message):
    with pytest.raises(RecipeError, match=message):
        validate_recipe(bad)


def test_validation_fills_defaults_and_orders_by_stage():
    out = validate_recipe(recipe(demand_cap={}, strait_open={"power": 2}, fraction={}))
    assert [e["atom"] for e in out["atoms"]] == [
        "strait_open",
        "fraction",
        "demand_cap",
    ]  # scale, scale (user order kept), stock
    assert out["atoms"][0]["params"] == {"power": 2}
    assert out["atoms"][2]["params"] == {"cover": 4.0, "slack": 1.5}


@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.json")))
def test_shipped_recipes_are_valid(path):
    validate_recipe(json.loads(path.read_text()))


# --- atoms on edited observations --------------------------------------------------------------------------------


def test_strait_open_scales_by_power(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    obs = copy.deepcopy(obs)
    obs["graph_now.open"][:] = 0.5
    obs["graph_now.open.observed"][:] = 1
    through = ctx.strait_matrix.sum(axis=1) > 0
    assert through.any()
    for power in (0.0, 1.0, 2.0):
        out = ATOMS["strait_open"](ctx, power=power).apply(fresh_plan(ctx), obs).flows
        np.testing.assert_allclose(out[through], ctx.capacity[through] * 0.5**power)
        np.testing.assert_allclose(out[~through], ctx.capacity[~through])


def test_strait_open_ignores_unobserved_straits(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    obs = copy.deepcopy(obs)
    obs["graph_now.open"][:] = 0.0
    obs["graph_now.open.observed"][:] = 0
    np.testing.assert_allclose(ATOMS["strait_open"](ctx).apply(fresh_plan(ctx), obs).flows, ctx.capacity)


def test_fraction_scales_everything(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    np.testing.assert_allclose(ATOMS["fraction"](ctx, share=0.25).apply(fresh_plan(ctx), obs).flows, ctx.capacity / 4)


def test_warning_cut_only_above_threshold(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    obs = copy.deepcopy(obs)
    atom = ATOMS["warning_cut"](ctx, threshold=1.0, cut=0.5)
    obs["warning.score"][:] = 0.0
    obs["warning.score.observed"][:] = 1
    np.testing.assert_allclose(atom.apply(fresh_plan(ctx), obs).flows, ctx.capacity)
    obs["warning.score"][atom.unit[0]] = 3.0
    out = atom.apply(fresh_plan(ctx), obs).flows
    on_strait = ctx.strait_matrix[:, 0] > 0
    np.testing.assert_allclose(out[on_strait], ctx.capacity[on_strait] * 0.5)
    np.testing.assert_allclose(out[~on_strait], ctx.capacity[~on_strait])


def test_demand_cap_zero_with_ample_stock_and_bounded_without(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    atom = ATOMS["demand_cap"](ctx, cover=2.0, slack=1.0)
    rich = copy.deepcopy(obs)
    rich["stock.qty"][:] = 1e12
    out = atom.apply(fresh_plan(ctx), rich).flows
    into_market = np.concatenate([atom.members[r] for r in range(len(atom.demands))])
    assert into_market.size and np.all(out[into_market] == 0.0)
    poor = copy.deepcopy(obs)
    poor["stock.qty"][:] = 0.0
    poor["backlog.qty"][:] = 0.0
    poor["pipeline.qty.observed"][:] = 0
    out = atom.apply(fresh_plan(ctx), poor).flows
    forecast = np.where(poor["demand_forecast.qty.observed"] == 1, poor["demand_forecast.qty"], 0.0)
    for r, idx in enumerate(atom.members):
        if idx.size:
            assert out[idx].sum() <= 2.0 * forecast[r, :2].mean() * (1 + 1e-9) + 1e-9


def test_sanction_frontload_raises_towards_capacity(tiny):
    _env, obs, config = tiny
    ctx = Context(config)
    slot = int(np.flatnonzero(ctx.capacity > 0)[0])
    edge = sorted(ctx.route_edges[slot])[0]
    obs = copy.deepcopy(obs)
    for key in ("edge", "k", "effective_week"):
        obs[f"pending_prohibitions.{key}.observed"][:] = 0
    obs["pending_prohibitions.edge"][0], obs["pending_prohibitions.k"][0] = edge, ctx.k[slot]
    obs["pending_prohibitions.effective_week"][0] = int(obs["week"][0]) + 2
    for key in ("edge", "k", "effective_week"):
        obs[f"pending_prohibitions.{key}.observed"][0] = 1
    obs["action_mask"][:] = 1
    out = ATOMS["sanction_frontload"](ctx, lead=3.0, boost=1.0).apply(fresh_plan(ctx, np.zeros(ctx.n_slots)), obs).flows
    assert out[slot] == pytest.approx(ctx.capacity[slot])
    far = ATOMS["sanction_frontload"](ctx, lead=1.0, boost=1.0).apply(fresh_plan(ctx, np.zeros(ctx.n_slots)), obs).flows
    assert far.sum() == 0.0  # takes effect in 2 weeks, beyond a lead of 1


# --- the guard ---------------------------------------------------------------------------------------------------


def make_agent(config, rec):
    namespace = {}
    exec(compile(modular_build.render(rec), "generated_agent.py", "exec"), namespace)
    return namespace["Agent"](config)


@pytest.mark.parametrize("fault", ["nan", "negative", "too_big", "raises", "wrong_shape"])
def test_guard_survives_a_faulty_atom(tiny, monkeypatch, fault):
    """Played through the repository's own Agent, so the test atom is in its registry."""
    import modular.core as core

    _env, obs, config = tiny

    @register
    class Faulty(Atom):
        name, stage = "faulty", "scale"

        def apply(self, plan, observation):
            if fault == "raises":
                raise RuntimeError("boom")
            plan.flows = {
                "nan": plan.flows * np.nan,
                "negative": -plan.flows - 1.0,
                "too_big": plan.flows * 1e6 + 1.0,
                "wrong_shape": plan.flows[:-1],
            }[fault]
            return plan

    try:
        monkeypatch.setattr(core, "RECIPE", recipe(faulty={}))
        agent = core.Agent(config)
        out = agent.act(obs)
        ctx = Context(config)
        mask = obs["action_mask"]
        assert out["flows"].shape == (ctx.n_slots,) and np.all(np.isfinite(out["flows"]))
        assert np.all(out["flows"] >= 0) and np.all(out["flows"] <= ctx.capacity * mask + 1e-9)
        assert out["release_mode"].dtype == np.int64
    finally:
        ATOMS.pop("faulty", None)


def test_flows_never_exceed_capacity_times_mask(tiny):
    env, obs, config = tiny
    agent = make_agent(config, json.loads((RECIPES / "full_stack.json").read_text()))
    ctx = Context(config)
    obs, _ = env.reset(options={"episode": 1})
    for _ in range(6):
        out = agent.act(obs)
        assert np.all(out["flows"] >= 0) and np.all(out["flows"] <= ctx.capacity * obs["action_mask"] + 1e-9)
        obs, *_ = env.step(out)


# --- the generated agent -----------------------------------------------------------------------------------------


def built(name, tmp_path):
    return Path(modular_build.build(RECIPES / f"{name}.json", tmp_path / name))


def test_generated_agent_is_self_contained(tmp_path):
    folder = built("full_stack", tmp_path)
    tree = ast.parse((folder / "agent.py").read_text())
    imported = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    assert imported <= {"json", "sys", "dataclasses", "numpy"}
    assert sorted(p.name for p in folder.iterdir()) == ["agent.py", "recipe.json"]


def test_build_refuses_to_overwrite_a_hand_written_agent(tmp_path):
    target = tmp_path / "mine"
    target.mkdir()
    (target / "agent.py").write_text("# hand written\n")
    with pytest.raises(RecipeError, match="not generated by modular.build"):
        modular_build.build(RECIPES / "empty.json", target)
    modular_build.build(RECIPES / "empty.json", target, force=True)


def test_empty_recipe_matches_template_and_heuristic_recipe_matches_heuristic(tiny, tmp_path):
    env, _obs, _config = tiny
    obs, info = env.reset(options={"episode": 2})
    config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    pairs = [("empty", agents.load("template")(config)), ("heuristic", agents.load("heuristic")(config))]
    generated = [agents.load(built(name, tmp_path))(config) for name, _ in pairs]
    done = False
    while not done:
        for g, (_name, ref) in zip(generated, pairs):
            np.testing.assert_allclose(g.act(obs)["flows"], ref.act(obs)["flows"], rtol=1e-12)
        obs, _r, te, tr, _i = env.step(pairs[1][1].act(obs))
        done = te or tr


def test_same_policy_seed_same_actions(tiny, tmp_path):
    env, _obs, _config = tiny
    folder = built("full_stack", tmp_path)

    def play():
        obs, info = env.reset(options={"episode": 3})
        agent = agents.load(folder)(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        trace, done = [], False
        while not done:
            out = agent.act(obs)
            trace.append(out["flows"].copy())
            obs, _r, te, tr, _i = env.step(out)
            done = te or tr
        return np.array(trace)

    np.testing.assert_array_equal(play(), play())


@pytest.mark.parametrize("name", ["empty", "heuristic", "closure_cap", "full_stack"])
def test_generated_agent_passes_sbf_check_on_small(name, tmp_path, capsys):
    sbf.check(str(built(name, tmp_path)), task="small")
    out = capsys.readouterr().out
    assert "all checks passed" in out and "WARNING" not in out


def test_param_is_a_frozen_description():
    assert Param(1.0, 0.0, 2.0).hi == 2.0
