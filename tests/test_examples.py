"""Every example runs on Tiny with small settings, in a subprocess, as `uv run python examples/<name>.py` runs it."""

import importlib.util
import json
import subprocess
import sys

import pytest

from tests.conftest import ROOT


EXAMPLES = ROOT / "examples"


def run(name: str, *args: str, env: dict, cwd, timeout: float = 300) -> str:
    """Run an example in ``cwd``; its output (stdout and stderr) when it exits 0."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, f"{name} {args} exited {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout + proc.stderr


def test_quickstart(env_with_cache, tmp_path):
    out = run("01_quickstart.py", env=env_with_cache, cwd=tmp_path)
    assert "26 weeks" in out and "options={'episode': 0}" in out and "USD with random actions" in out


def test_play_agents(env_with_cache, tmp_path):
    out = run("02_play_agents.py", "--episodes=1", env=env_with_cache, cwd=tmp_path)
    assert "random" in out and "template" in out and "USD" in out


def test_heuristic_agent(env_with_cache, tmp_path):
    out = run("03_heuristic_agent.py", "--episodes=1", env=env_with_cache, cwd=tmp_path)
    assert "a strait falls below 50% open: [29]" in out
    assert "in total the rule saves " in out and "in total the rule saves -" not in out  # it acts where it closes


def test_evaluate_and_compare(env_with_cache, tmp_path):
    out = run("04_evaluate.py", "--quick", "--n_jobs=1", env=env_with_cache, cwd=tmp_path)
    assert "Score:" in out and "interval" in out and "quick:" in out
    agent_file = ROOT / "agents" / "heuristic" / "agent.py"
    args = [f"--agent={agent_file}", "--against=template", "--quick", "--n_jobs=1"]
    out = run("04_evaluate.py", *args, env=env_with_cache, cwd=tmp_path)
    assert "A - B:" in out and "paired interval" in out


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="the rl extra is not installed")
def test_train_ppo(env_with_cache, tmp_path):
    args = ["--total_timesteps=416", "--n_envs=1", "--n_scenarios=2", "--check_episodes=1", "--activation=relu"]
    args += ["--net_arch=[16]", "--out=run"]
    out = run("05_train_ppo.py", *args, env=env_with_cache, cwd=tmp_path)
    folder = tmp_path / "run"
    assert (folder / "submission.zip").is_file() and (folder / "submission" / "policy.pt").is_file()
    assert "POLICY = torch.jit.load" in (folder / "submission" / "agent.py").read_text()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["settings"]["net_arch"] == [16] and "max_flow_difference" in out


def test_policy_search(env_with_cache, tmp_path):
    args = ["--generations=1", "--population=2", "--train_episodes=2", "--quick", "--n_jobs=1", "--out=run"]
    out = run("06_policy_search.py", *args, env=env_with_cache, cwd=tmp_path)
    assert "held out, on 4 dev episodes" in out and "A - B:" in out
    best = tmp_path / "run" / "best"
    assert best.is_dir() == ("not written" not in out)  # written only when it beats send-the-maximum held out
    if best.is_dir():
        params = json.loads((best / "params.json").read_text())
        assert len(params["fraction"]) == 20 and (best / "agent.py").is_file()


def test_dashboard(env_with_cache, tmp_path):
    output = run(
        "07_dashboard.py", "--episode=29", "--quick", "--n_jobs=1", "--out=run", env=env_with_cache, cwd=tmp_path
    )
    assert "mine minus template:" in output
    for name in (
        "network.png",
        "dashboard_mine.png",
        "dashboard_max.png",
        "episode_mine.gif",
        "episode_max.gif",
        "record_mine.npz",
        "record_max.npz",
    ):
        assert (tmp_path / "run" / name).is_file(), name


def test_compare_training_costs(env_with_cache, tmp_path):
    run("08_compare_costs.py", "--episodes=1", "--out=run", env=env_with_cache, cwd=tmp_path)
    rows = json.loads((tmp_path / "run" / "episodes.json").read_text())
    assert [row["agent"] for row in rows] == ["baseline", "mine"]
    assert all(row["entropy"] == 12345 and row["cost_usd"] > 0 for row in rows)
    assert all(abs(sum(row["components"].values()) - row["salvage_usd"] - row["cost_usd"]) < 1 for row in rows)
