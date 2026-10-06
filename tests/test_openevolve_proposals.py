"""Qwen proposals must change numeric behavior and retry repeats before simulation."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def proposal_module(monkeypatch):
    folder = Path(__file__).parents[1] / "examples/10_openevolve"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("sbf_numeric_proposals", folder / "proposal_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(proposal_module, tmp_path, monkeypatch):
    params = {"reserve_weeks": 4.0, "discount": 0.9, "allocation_blend": 0.7}
    manifest = {
        "strategy": "constant",
        "ollama": "http://localhost:11434",
        "proposal_config": {
            "temperature": 0.95,
            "top_p": 0.95,
            "attempts": 3,
            "min_change": 0.03,
            "context": 8192,
            "seed": 100,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "baseline-evaluation.json").write_text(json.dumps({"cost_cents": [100]}))
    seed = tmp_path / "candidates/seed"
    seed.mkdir(parents=True)
    (seed / "params.json").write_text(json.dumps(params))
    (seed / "evaluation.json").write_text(
        json.dumps({"cost_cents": [100], "fallback_weeks": 0, "cpu_weeks": 0, "invalid_entries": 0})
    )
    monkeypatch.setenv("SBF_EVOLVE_RUN", str(tmp_path))
    return proposal_module.NumericProposalClient(SimpleNamespace(name="qwen2.5-coder:3b")), params


def message(params):
    return [{"role": "user", "content": "# Current Program\nPARAMS = " + repr(params)}]


def response(params):
    return {"message": {"content": json.dumps(params)}, "prompt_eval_count": 10, "eval_count": 5}


def test_parent_extraction_uses_current_code_not_repeated_inspiration(proposal_module):
    text = "PARAMS = {'reserve_weeks': 1, 'discount': 0.7, 'allocation_blend': 0.1}\n"
    text += "# Current Program\nPARAMS = {'reserve_weeks': 4, 'discount': 0.9, 'allocation_blend': 0.7}"
    assert proposal_module.extract_parent([{"content": text}], "constant")["reserve_weeks"] == 4


def test_identical_numbers_and_tiny_steps_are_rejected(proposal_module):
    parent = {"reserve_weeks": 4.0, "discount": 0.9, "allocation_blend": 0.7}
    keys = list(parent)
    with pytest.raises(proposal_module.ProposalRejected, match="already proposed"):
        proposal_module.validate_patch(parent, dict(reversed(list(parent.items()))), keys, [], "constant", 0.03)
    with pytest.raises(proposal_module.ProposalRejected, match="Too close"):
        proposal_module.validate_patch(parent, {**parent, "reserve_weeks": 4.01}, keys, [], "constant", 0.03)


def test_rejects_old_candidate_even_when_it_differs_from_parent(proposal_module):
    parent = {"reserve_weeks": 4.0, "discount": 0.9, "allocation_blend": 0.7}
    old = {**parent, "reserve_weeks": 6.0}
    with pytest.raises(proposal_module.ProposalRejected):
        proposal_module.validate_patch(parent, old, list(parent), [old], "constant", 0.03)


@pytest.mark.parametrize("value", [True, float("inf"), "0.9", -1])
def test_patch_cannot_bypass_literal_parameter_validation(proposal_module, value):
    parent = {"reserve_weeks": 4.0, "discount": 0.9, "allocation_blend": 0.7}
    with pytest.raises(proposal_module.ProposalRejected):
        proposal_module.validate_patch(parent, {**parent, "discount": value}, list(parent), [], "constant", 0.03)


def test_duplicate_gets_feedback_retry_and_new_seed(client, proposal_module, monkeypatch):
    model, parent = client
    calls = []

    def post(payload):
        calls.append(payload)
        return response(parent if len(calls) == 1 else {**parent, "reserve_weeks": 6.0})

    monkeypatch.setattr(model, "_post", post)
    code = asyncio.run(model.generate_with_context("generic rewrite prompt", message(parent)))
    assert proposal_module.read_params(code)["reserve_weeks"] == 6
    assert len(calls) == 2
    assert "Previous proposal rejected" in calls[1]["messages"][-1]["content"]
    assert calls[0]["options"]["seed"] != calls[1]["options"]["seed"]
    assert calls[0]["format"]["additionalProperties"] is False
    assert calls[0]["options"]["num_ctx"] == 8192
    assert model.last_usage["total_tokens"] == 30
    stats = proposal_module.proposal_stats(model.run)
    assert stats["accepted_model_proposals"] == 1 and stats["rejected_duplicates"] == 1


def test_exhaustion_does_not_silently_mutate_or_claim_a_candidate(client, proposal_module, monkeypatch):
    model, parent = client
    monkeypatch.setattr(model, "_post", lambda payload: response(parent))
    with pytest.raises(RuntimeError, match="no new valid parameters"):
        asyncio.run(model.generate_with_context("", message(parent)))
    stats = proposal_module.proposal_stats(model.run)
    assert stats["accepted_model_proposals"] == 0
    assert stats["rejected_duplicates"] == 3 and stats["exhausted_proposals"] == 1


def test_accepted_proposal_is_remembered_before_benchmark_writes_it(client, proposal_module, monkeypatch):
    model, parent = client
    alternative = {**parent, "reserve_weeks": 6.0}
    calls = []

    def post(payload):
        calls.append(payload)
        return response(alternative if len(calls) <= 2 else {**parent, "reserve_weeks": 2.0})

    monkeypatch.setattr(model, "_post", post)
    asyncio.run(model.generate_with_context("", message(parent)))
    code = asyncio.run(model.generate_with_context("", message(parent)))
    assert proposal_module.read_params(code)["reserve_weeks"] == 2
    assert proposal_module.proposal_stats(model.run)["accepted_model_proposals"] == 2
