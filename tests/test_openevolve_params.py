"""Model output must remain literal bounded parameters, without executing code."""

import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / "examples" / "10_openevolve" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("sbf_openevolve_evaluator", MODULE)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)
VALID = 'PARAMS = {"reserve_weeks": 3, "discount": 0.85, "allocation_blend": 0.5}'


def test_seed_matches_mine_defaults():
    seed = MODULE.with_name("seed.py").read_text()
    assert EVALUATOR.read_params(seed) == EVALUATOR.read_params(VALID)


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + VALID,
        VALID + "\nprint('executed')",
        "PARAMS = dict(reserve_weeks=3, discount=0.85, allocation_blend=0.5)",
        VALID.replace("PARAMS", "other"),
        VALID.replace('"discount": 0.85', '"discount": 0'),
        VALID.replace('"discount": 0.85', '"discount": 1.1'),
        VALID.replace('"reserve_weeks": 3', '"reserve_weeks": -1'),
        VALID.replace('"reserve_weeks": 3', '"reserve_weeks": 9'),
        VALID.replace('"allocation_blend": 0.5', '"allocation_blend": True'),
        VALID.replace('"allocation_blend": 0.5', '"allocation_blend": 1e999'),
        VALID.replace('"allocation_blend": 0.5', '"allocation_blend": "0.5"'),
        VALID.replace('"allocation_blend": 0.5', '"unknown": 0.5'),
        VALID + "\nPARAMS = {}",
    ],
)
def test_rejects_invalid_or_executable_output(code):
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        EVALUATOR.read_params(code)


def test_candidate_identity_ignores_mapping_order():
    params = EVALUATOR.read_params(VALID)
    assert EVALUATOR.candidate_id(params) == EVALUATOR.candidate_id(dict(reversed(list(params.items()))))


def test_adaptive_schema_matches_agent_and_zero_seed():
    from agents.adaptive_reserve import agent

    assert set(agent.PARAMS) == set(EVALUATOR.ADAPTIVE_BOUNDS)
    for key, bounds in agent.EXTENSION_BOUNDS.items():
        assert bounds == EVALUATOR.ADAPTIVE_BOUNDS[key]
    params = EVALUATOR.read_params(f"PARAMS = {agent.PARAMS!r}", "adaptive")
    assert all(value == 0 for key, value in params.items() if key.endswith("_weight"))


@pytest.mark.parametrize("key", list(EVALUATOR.ADAPTIVE_BOUNDS))
def test_adaptive_rejects_out_of_range_parameters(key):
    from agents.adaptive_reserve import agent

    params = dict(agent.PARAMS)
    params[key] = EVALUATOR.ADAPTIVE_BOUNDS[key][1] + 1
    with pytest.raises(ValueError):
        EVALUATOR.read_params(f"PARAMS = {params!r}", "adaptive")
