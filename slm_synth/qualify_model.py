"""Qualify an OpenRouter model for the five supported synthetic datasets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.model_suitability import ModelSuitabilityError, get_reasoning_suitability
from slm_synth.runtime import build_backend

DATASET_ROLES: dict[str, tuple[str, ...]] = {
    "pretrain": ("pretrain-generator", "pretrain-judge", "pretrain-reviewer"),
    "sft": ("sft-generator", "sft-judge", "sft-reviewer"),
    "dpo": ("dpo-generator", "dpo-judge", "dpo-reviewer"),
    "distillation-sft": (
        "distillation-sft-generator",
        "distillation-sft-judge",
        "distillation-sft-reviewer",
    ),
    "distillation-dpo": (
        "distillation-dpo-generator",
        "distillation-dpo-judge",
        "distillation-dpo-reviewer",
    ),
}
ROLES = frozenset(role for roles in DATASET_ROLES.values() for role in roles)


def resolve_roles(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("at least one qualification role or dataset is required")
    if len(requested) == 1 and requested[0].lower() == "all":
        return sorted(ROLES)

    roles: list[str] = []
    for item in requested:
        if item in DATASET_ROLES:
            roles.extend(DATASET_ROLES[item])
        elif item in ROLES:
            roles.append(item)
        else:
            raise ValueError(f"unknown qualification role or dataset: {item!r}")
    return list(dict.fromkeys(roles))


def _generator_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"content": {"type": "string", "minLength": 1}},
                    "required": ["content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _standard_judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    "probe": {
                        "type": "object",
                        "properties": {
                            "assessable": {"type": "boolean"},
                            "accepted": {"type": "boolean"},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["assessable", "accepted", "reason"],
                        "additionalProperties": False,
                    }
                },
                "required": ["probe"],
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _pretrain_judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    "probe": {
                        "type": "object",
                        "properties": {
                            "assessable": {"type": "boolean"},
                            "quality_valid": {"type": "boolean"},
                            "signal_aligned": {"type": "boolean"},
                            "natural_and_useful": {"type": "boolean"},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "assessable",
                            "quality_valid",
                            "signal_aligned",
                            "natural_and_useful",
                            "reason",
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["probe"],
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _distillation_dpo_judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    "probe": {
                        "type": "object",
                        "properties": {
                            "assessable": {"type": "boolean"},
                            "chosen_complete": {"type": "boolean"},
                            "chosen_correct": {"type": "boolean"},
                            "preference_valid": {"type": "boolean"},
                            "dimension_aligned": {"type": "boolean"},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "assessable",
                            "chosen_complete",
                            "chosen_correct",
                            "preference_valid",
                            "dimension_aligned",
                            "reason",
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["probe"],
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _reviewer_schema(role: str) -> dict[str, Any]:
    if role == "pretrain-reviewer":
        decision = {
            "type": "object",
            "properties": {
                "reviewed": {"type": "boolean"},
                "reviewer_agreed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["reviewed", "reviewer_agreed", "reason"],
            "additionalProperties": False,
        }
    else:
        decision = {
            "type": "object",
            "properties": {
                "agreed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["agreed", "reason"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {"probe": decision},
                "required": ["probe"],
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _probe(role: str) -> tuple[str, dict[str, Any], str]:
    if role.endswith("generator"):
        return (
            "Return exactly one concise content item grounded only in this instruction.",
            _generator_schema(),
            "qualification_generator",
        )
    if role == "pretrain-judge":
        return (
            "Assess probe. It asks for an unsupported private fact with no evidence. Mark every quality gate false and explain briefly.",
            _pretrain_judge_schema(),
            "qualification_pretrain_judge",
        )
    if role == "distillation-dpo-judge":
        return (
            "Assess probe. The task is unsupported and the chosen/rejected responses cannot be meaningfully compared. Mark every gate false and explain briefly.",
            _distillation_dpo_judge_schema(),
            "qualification_distillation_dpo_judge",
        )
    if role.endswith("judge"):
        return (
            "Assess probe. The task asks for an unsupported private fact with no evidence. Set assessable=false, accepted=false, and explain briefly.",
            _standard_judge_schema(),
            "qualification_judge",
        )
    return (
        "Review probe. The judge accepted an unsupported invented claim. Disagree with the judge and explain briefly.",
        _reviewer_schema(role),
        "qualification_reviewer",
    )


def _behavior_passed(role: str, data: dict[str, Any]) -> bool:
    if role.endswith("generator"):
        items = data.get("items")
        return (
            isinstance(items, list)
            and len(items) == 1
            and isinstance(items[0], dict)
            and isinstance(items[0].get("content"), str)
            and bool(items[0]["content"].strip())
        )

    decisions = data.get("decisions")
    decision = decisions.get("probe") if isinstance(decisions, dict) else None
    if not isinstance(decision, dict):
        return False
    if role == "pretrain-judge":
        return all(
            decision.get(field) is False
            for field in ("assessable", "quality_valid", "signal_aligned", "natural_and_useful")
        )
    if role == "distillation-dpo-judge":
        return all(
            decision.get(field) is False
            for field in (
                "assessable",
                "chosen_complete",
                "chosen_correct",
                "preference_valid",
                "dimension_aligned",
            )
        )
    if role.endswith("judge"):
        return decision.get("assessable") is False and decision.get("accepted") is False
    if role == "pretrain-reviewer":
        return decision.get("reviewed") is True and decision.get("reviewer_agreed") is False
    return decision.get("agreed") is False


def qualify(
    *,
    model: str,
    roles: list[str],
    max_tokens: int,
    routing_mode: str,
    provider: str | None = None,
    behavior_trials: int = 3,
) -> dict[str, Any]:
    if behavior_trials < 1:
        raise ValueError("behavior_trials must be >= 1")
    unknown = sorted(set(roles) - ROLES)
    if unknown:
        raise ValueError(f"unknown qualification role(s): {unknown}")

    try:
        reasoning_suitability = get_reasoning_suitability(model)
        reasoning_policy = reasoning_suitability.as_dict()
    except ModelSuitabilityError as exc:
        reasoning_policy = {
            "model": model,
            "reasoning_capable": None,
            "reasoning_mandatory": None,
            "reasoning_disable_supported": False,
            "reasoning_policy_pass": False,
            "source": "OpenRouter",
            "error": str(exc),
        }

    if not reasoning_policy["reasoning_policy_pass"]:
        results = {
            role: {
                "transport_compatible": None,
                "contract_pass": False,
                "behavioral_pass": False,
                "reasoning_policy_pass": False,
                "passed": False,
                "error": reasoning_policy.get("error")
                or "Model cannot run with reasoning disabled",
            }
            for role in roles
        }
        return {
            "schema_version": 6,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "contract": "openrouter_strict_json_schema_v1",
            "reasoning_policy": reasoning_policy,
            "behavior_trials": behavior_trials,
            "roles": results,
            "passed": False,
        }

    reasoning_requires_live_verification = (
        reasoning_policy.get("reasoning_capable") is True
        and reasoning_policy.get("reasoning_mandatory") is False
        and reasoning_policy.get("reasoning_disable_supported") is None
    )
    reasoning_mandatory = reasoning_policy.get("reasoning_mandatory") is True
    if reasoning_mandatory:
        reasoning_policy["reasoning_disable_verified"] = False
        reasoning_policy["verification"] = "not_applicable_mandatory_reasoning"
    else:
        reasoning_policy["reasoning_disable_verified"] = not reasoning_requires_live_verification
        reasoning_policy["verification"] = (
            "not_required"
            if not reasoning_requires_live_verification
            else "pending_live_request"
        )

    try:
        backend = build_backend(
            model=model,
            max_tokens=max_tokens,
            concurrency=1,
            routing_mode=routing_mode,
            provider=provider,
            temperature=None,
            top_p=None,
        )
    except Exception as exc:
        results = {
            role: {
                "transport_compatible": False,
                "contract_pass": False,
                "behavioral_pass": False,
                "reasoning_policy_pass": False,
                "passed": False,
                "error": str(exc),
            }
            for role in roles
        }
        reasoning_policy["reasoning_policy_pass"] = False
        reasoning_policy["reasoning_disable_verified"] = False
        reasoning_policy["verification"] = "backend_construction_failed"
        return {
            "schema_version": 6,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "contract": "openrouter_strict_json_schema_v1",
            "reasoning_policy": reasoning_policy,
            "behavior_trials": behavior_trials,
            "roles": results,
            "passed": False,
        }

    results: dict[str, Any] = {}
    for role in roles:
        prompt, schema, schema_name = _probe(role)
        trials: list[dict[str, Any]] = []
        for trial_number in range(1, behavior_trials + 1):
            try:
                envelope = backend.generate_structured_object_with_metadata(
                    prompt=prompt,
                    schema=schema,
                    schema_name=schema_name,
                )
                data = envelope.get("data") if isinstance(envelope, dict) else None
                telemetry = envelope.get("telemetry") if isinstance(envelope, dict) else None
                contract_pass = isinstance(data, dict)
                behavioral_pass = contract_pass and _behavior_passed(role, data)
                trials.append(
                    {
                        "trial": trial_number,
                        "transport_compatible": True,
                        "contract_pass": contract_pass,
                        "behavioral_pass": behavioral_pass,
                        "passed": bool(contract_pass and behavioral_pass),
                        "response": data,
                        "telemetry": telemetry if isinstance(telemetry, dict) else {},
                    }
                )
            except Exception as exc:
                trials.append(
                    {
                        "trial": trial_number,
                        "transport_compatible": False,
                        "contract_pass": False,
                        "behavioral_pass": False,
                        "passed": False,
                        "error": str(exc),
                    }
                )

        passed_trials = sum(1 for trial in trials if trial["passed"])
        role_result = {
            "transport_compatible": all(trial["transport_compatible"] is True for trial in trials),
            "contract_pass": all(trial["contract_pass"] is True for trial in trials),
            "behavioral_pass": all(trial["behavioral_pass"] is True for trial in trials),
            "reasoning_policy_pass": True,
            "passed": passed_trials == behavior_trials,
            "behavior_trials_requested": behavior_trials,
            "behavior_trials_passed": passed_trials,
            "trials": trials,
        }
        first_error = next((trial.get("error") for trial in trials if trial.get("error")), None)
        if first_error is not None:
            role_result["error"] = first_error
        results[role] = role_result

    if reasoning_requires_live_verification:
        reasoning_disable_verified = any(
            trial.get("transport_compatible") is True
            for result in results.values()
            for trial in result.get("trials", [])
        )
        reasoning_policy["reasoning_disable_supported"] = reasoning_disable_verified
        reasoning_policy["reasoning_disable_verified"] = reasoning_disable_verified
        reasoning_policy["reasoning_policy_pass"] = reasoning_disable_verified
        reasoning_policy["verification"] = (
            "live_strict_json_request_with_reasoning_effort_none"
            if reasoning_disable_verified
            else "live_request_failed"
        )
        for result in results.values():
            result["reasoning_policy_pass"] = reasoning_disable_verified
            result["passed"] = bool(result["passed"] and reasoning_disable_verified)

    return {
        "schema_version": 6,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "contract": "openrouter_strict_json_schema_v1",
        "reasoning_policy": reasoning_policy,
        "behavior_trials": behavior_trials,
        "roles": results,
        "passed": bool(
            reasoning_policy["reasoning_policy_pass"]
            and all(result["passed"] for result in results.values())
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--roles",
        required=True,
        help=(
            "Comma-separated role names, one or more dataset names "
            f"({', '.join(DATASET_ROLES)}), or all"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--behavior-trials",
        type=int,
        default=int(os.getenv("QUALIFY_BEHAVIOR_TRIALS", "3")),
        help="Independent behavioral qualification trials required per role (default: 3)",
    )
    parser.add_argument(
        "--openrouter-routing-mode",
        choices=["auto", "prefer", "strict"],
        default="auto",
    )
    parser.add_argument("--openrouter-provider", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roles = resolve_roles(args.roles)
    report = qualify(
        model=args.model,
        roles=roles,
        max_tokens=args.max_tokens,
        routing_mode=args.openrouter_routing_mode,
        provider=args.openrouter_provider,
        behavior_trials=args.behavior_trials,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote model qualification report to {path}")
    else:
        print(rendered, end="")

    qualification = "PASS" if report["passed"] else "FAIL"
    print(f"model qualification result: {qualification}")

    strict = os.getenv("QUALIFY_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return 1 if strict and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
