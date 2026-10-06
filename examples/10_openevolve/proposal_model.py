"""Ask Ollama for bounded numeric changes and reject semantic repeats before play."""

import asyncio
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from evaluator import BOUNDS, candidate_id, read_params, strategy_bounds, training_savings


LOGGER = logging.getLogger(__name__)
BLOCKS = [
    ["reserve_weeks", "delay_weight", "spread_weight", "shortage_weight"],
    ["lp_shortage_weight", "lp_delay_weight", "allocation_blend"],
    ["lng_terminal_weight", "lng_ration_weight", "lng_reserve_weight", "lng_lp_weight"],
    ["semi_allocation_weight", "semi_shortage_weight", "semi_value_weight", "semi_delay_weight"],
    ["discount", "reserve_min", "reserve_max"],
]
MPC_BLOCKS = [
    ["planning_horizon", "eta_margin", "supply_factor"],
    ["demand_factor", "terminal_weeks", "terminal_weight"],
    ["mpc_blend", "change_penalty", "discount"],
    ["reserve_weeks", "allocation_blend", "mpc_blend"],
]
MEANINGS = {
    "planning_horizon": "MPC weekly horizon, rounded to nearest integer",
    "mpc_blend": "share of time-expanded MPC plan versus frozen v8 action",
    "eta_margin": "extra weeks on public delivery ETA for MPC",
    "supply_factor": "multiplier on observed source refill forecast",
    "demand_factor": "multiplier on observed grid fuel consumption forecast",
    "terminal_weeks": "desired grid inventory coverage at horizon end",
    "terminal_weight": "weight of horizon-end fuel shortage relative to blackout penalty",
    "change_penalty": "penalty on absolute difference from current v8 request",
    "reserve_weeks": "base fuel reserve in weeks",
    "delay_weight": "extra reserve per week of excess delivery delay; negative reduces reserve",
    "spread_weight": "extra reserve per week of difference between route ETAs",
    "shortage_weight": "extra reserve per week of fuel deficit before first arrival",
    "lp_shortage_weight": "LP share offset when fuel shortage increases",
    "lp_delay_weight": "LP share offset when delivery delay increases",
    "allocation_blend": "base LP share versus template shipments",
    "lng_terminal_weight": "0 pooled LNG inventory; 1 grid-accessible LNG delivery forecast",
    "lng_ration_weight": "fraction of the grid's LNG rationing stock floor",
    "lng_reserve_weight": "additional reserve weeks for LNG only",
    "lng_lp_weight": "LP share offset for LNG only",
    "semi_allocation_weight": "0 template; 1 allocation of scarce semiconductor stock by priority",
    "semi_shortage_weight": "positive favors unmet factory or market needs",
    "semi_value_weight": "positive favors higher economic value",
    "semi_delay_weight": "positive favors shorter semiconductor delivery times",
    "discount": "discount factor for future fuel-shortage penalties",
    "reserve_min": "minimum fuel reserve in weeks",
    "reserve_max": "maximum fuel reserve in weeks; must be >= reserve_min",
}


class ProposalRejected(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def parameter_distance(a, b, bounds):
    """Largest change as a fraction of a parameter's allowed range."""
    return max(abs(a[key] - b[key]) / (high - low) for key, (low, high) in bounds.items())


def validate_patch(parent, patch, keys, seen, strategy, min_change):
    if not isinstance(patch, dict) or set(patch) != set(keys):
        raise ProposalRejected("invalid", f"Return exactly these keys: {keys}")
    try:
        params = read_params(f"PARAMS = {dict(parent, **patch)!r}", strategy)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ProposalRejected("invalid", str(exc)) from exc
    bounds = strategy_bounds(strategy)
    for previous in [parent, *seen]:
        distance = parameter_distance(params, previous, bounds)
        if distance < 1e-12:
            raise ProposalRejected("duplicate", "These numeric values were already proposed. Change the values.")
        if distance < min_change:
            raise ProposalRejected(
                "too_close",
                f"Too close to an earlier set: change at least one value by {min_change:.0%} of its allowed range.",
            )
    return params


def extract_parent(messages, strategy):
    text = "\n".join(message["content"] for message in messages)
    # OpenEvolve's rewrite prompt repeats inspiration code. Use the current parent.
    text = text.split("# Current Program")[-1]
    match = re.search(r"PARAMS\s*=\s*(\{.*?\})", text, re.S)
    if not match:
        raise ValueError("OpenEvolve prompt does not contain the current PARAMS mapping")
    return read_params("PARAMS = " + match.group(1), strategy)


def proposal_stats(run):
    journal = Path(run) / "proposals.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    return {
        "model_calls": sum(event["status"] != "exhausted" for event in events),
        "accepted_model_proposals": sum(event["status"] == "accepted" for event in events),
        "rejected_duplicates": sum(event.get("reason") == "duplicate" for event in events),
        "rejected_too_close": sum(event.get("reason") == "too_close" for event in events),
        "rejected_invalid": sum(event.get("reason") == "invalid" for event in events),
        "exhausted_proposals": sum(event["status"] == "exhausted" for event in events),
    }


class NumericProposalClient:
    """OpenEvolve LLM interface backed by Ollama JSON-schema generation.

    Every accepted number comes from the model. No random mutation fallback.
    Serial OpenEvolve evaluation keeps this run-local journal and admission atomic.
    """

    def __init__(self, model_cfg):
        self.model = model_cfg.name
        self.run = Path(os.environ["SBF_EVOLVE_RUN"])
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        self.strategy = self.manifest.get("strategy", "constant")
        self.settings = self.manifest["proposal_config"]
        self.bounds = strategy_bounds(self.strategy)
        self.last_usage = None

    def _events(self):
        file = self.run / "proposals.jsonl"
        return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []

    def _record(self, event):
        with (self.run / "proposals.jsonl").open("a") as stream:
            stream.write(json.dumps({"source": "qwen", **event}) + "\n")

    def _history(self):
        seen, measured = [], []
        baseline = json.loads((self.run / "baseline-evaluation.json").read_text())
        for folder in sorted((self.run / "candidates").glob("*")):
            params = json.loads((folder / "params.json").read_text())
            seen.append(params)
            file = folder / "evaluation.json"
            if file.exists():
                result = json.loads(file.read_text())
                issues = sum(result[key] for key in ("fallback_weeks", "cpu_weeks", "invalid_entries"))
                score = -1e6 if issues else training_savings(result, baseline)[0]
                measured.append({"params": params, "cost_saving_percent": score})
        seen.extend(event["params"] for event in self._events() if event["status"] == "accepted")
        return seen, sorted(measured, key=lambda row: -row["cost_saving_percent"])[:4]

    def _post(self, payload):
        request = urllib.request.Request(
            self.manifest["ollama"].rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message, messages, **kwargs):
        parent = extract_parent(messages, self.strategy)
        seen, measured = self._history()
        previous_events = self._events()
        proposal = 1 + sum(event["status"] in ("accepted", "exhausted") for event in previous_events)
        blocks = MPC_BLOCKS if self.strategy == "mpc" else BLOCKS
        block = blocks[(proposal - 1) % len(blocks)] if self.strategy != "constant" else list(BOUNDS)
        offset = ((proposal - 1) // len(blocks)) % len(block)
        keys = (block[offset:] + block[:offset])[:3]
        schema = {
            "type": "object",
            "properties": {
                key: {"type": "number", "minimum": self.bounds[key][0], "maximum": self.bounds[key][1]} for key in keys
            },
            "required": keys,
            "additionalProperties": False,
        }
        history = [
            {"values": {key: row["params"][key] for key in keys}, "cost_saving_percent": row["cost_saving_percent"]}
            for row in measured
        ]
        feedback = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": self.model}
        for attempt in range(1, self.settings["attempts"] + 1):
            call = 1 + sum(event["status"] != "exhausted" for event in self._events())
            seed = self.settings["seed"] + call
            temperature = min(1.2, self.settings["temperature"] + 0.05 * (attempt - 1))
            descriptions = {key: {"bounds": self.bounds[key], "meaning": MEANINGS[key]} for key in keys}
            user = (
                f"Experiment {proposal}, attempt {attempt}. Current selected values: "
                f"{json.dumps({key: parent[key] for key in keys})}. "
                f"Bounds and meanings: {json.dumps(descriptions)}. "
                f"Measured training results (higher saving is better): {json.dumps(history)}. "
                f"Change at least one selected value by {self.settings['min_change']:.0%} of its allowed range. "
                "Explore a materially different experiment, including negative weights where allowed. "
                "Choose your own values. Return only the selected keys as JSON numbers. "
                "All other agent parameters stay fixed. reserve_min must not exceed reserve_max. " + feedback
            )
            payload = {
                "model": self.model,
                "stream": False,
                "format": schema,
                "keep_alive": "15m",
                "options": {
                    "temperature": temperature,
                    "top_p": self.settings["top_p"],
                    "repeat_penalty": 1.05,
                    "num_ctx": self.settings["context"],
                    "num_predict": 256,
                    "seed": seed,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You design numerical experiments for a supply-network policy. "
                            "Use measured costs to propose new parameter values. "
                            "Copying previous numbers is rejected. Respond only with a JSON object."
                        ),
                    },
                    {"role": "user", "content": user},
                ],
            }
            event = {"proposal": proposal, "attempt": attempt, "keys": keys, "seed": seed, "temperature": temperature}
            try:
                response = await asyncio.to_thread(self._post, payload)
                prompt_tokens = int(response.get("prompt_eval_count", 0))
                completion_tokens = int(response.get("eval_count", 0))
                usage["prompt_tokens"] += prompt_tokens
                usage["completion_tokens"] += completion_tokens
                usage["total_tokens"] += prompt_tokens + completion_tokens
                patch = json.loads(response["message"]["content"])
                event["patch"] = patch
                params = validate_patch(parent, patch, keys, seen, self.strategy, self.settings["min_change"])
            except ProposalRejected as exc:
                self._record({**event, "status": "rejected", "reason": exc.reason, "error": str(exc)})
                feedback = (
                    f"Previous proposal rejected: {exc}. Previous JSON: {json.dumps(event.get('patch'))}. "
                    "Try a different magnitude or sign."
                )
                LOGGER.info("Qwen proposal %s rejected (%s); retrying", proposal, exc.reason)
                continue
            except (ValueError, KeyError, OSError) as exc:
                self._record({**event, "status": "rejected", "reason": "invalid", "error": str(exc)})
                feedback = f"Previous response failed validation: {exc}. Return valid JSON within bounds."
                continue
            self.last_usage = usage
            self._record(
                {
                    **event,
                    "status": "accepted",
                    "params": params,
                    "candidate_id": candidate_id(params),
                    "parent_distance": parameter_distance(params, parent, self.bounds),
                }
            )
            LOGGER.info("Qwen proposal %s accepted: %s", proposal, patch)
            return "# EVOLVE-BLOCK-START\nPARAMS = " + repr(params) + "\n# EVOLVE-BLOCK-END\n"
        self.last_usage = usage
        self._record({"proposal": proposal, "status": "exhausted"})
        raise RuntimeError(
            f"Qwen produced no new valid parameters after {self.settings['attempts']} attempts; see proposals.jsonl"
        )


def create_proposal_client(model_cfg):
    return NumericProposalClient(model_cfg)
