"""Qualify an OpenRouter model against portable repository roles."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.llm import LLMBackend
from slm_synth.model_suitability import ModelSuitabilityError, get_reasoning_suitability
from slm_synth.model_contract import (
    PlainOutputContractError,
    call_plain_parsed,
    parse_json_object,
    parse_judge_decision,
    parse_review_decision,
)

ROLES = frozenset(
    {
        "pretrain-generator",
        "sft-generator",
        "sft-judge",
        "sft-reviewer",
        "dpo-generator",
        "dpo-judge",
        "dpo-reviewer",
        "distillation-sft-generator",
        "distillation-sft-judge",
        "distillation-sft-reviewer",
        "distillation-dpo-generator",
        "distillation-dpo-judge",
        "distillation-dpo-reviewer",
    }
)


def _probe(role: str) -> tuple[str, str, Any]:
    if role.endswith("generator"):
        prompt = (
            "Return exactly one JSON object with this shape: "
            '{"items":[{"content":"A concise answer grounded only in the prompt."}]}. '
            "Do not add metadata, Markdown, or commentary."
        )
        return "You generate only requested language-bearing dataset content.", prompt, parse_json_object
    if role.endswith("judge"):
        prompt = (
            "The task asks for the exact cause of an unnamed Atlas project delay and supplies no organization, date, or "
            "source. Reject because it is not reliably assessable. Return exactly:\n"
            "ASSESSABLE: YES or NO\n"
            "DECISION: ACCEPT or REJECT\nREASON: one concise reason in English"
        )
        return (
            "You are a conservative dataset judge. Never guess. Respond in English.",
            prompt,
            parse_judge_decision,
        )
    prompt = (
        "A judge accepted an answer that invented the cause of an unnamed Atlas project delay despite no evidence. "
        "State whether you agree. Return exactly:\nAGREE: YES or NO\n"
        "REASON: one concise reason in English"
    )
    return (
        "You independently review dataset judge decisions. Respond in English.",
        prompt,
        parse_review_decision,
    )


def _generator_passed(value: Any) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"items"}
        or not isinstance(value["items"], list)
    ):
        return False
    if len(value["items"]) != 1 or not isinstance(value["items"][0], dict):
        return False
    item = value["items"][0]
    return set(item) == {"content"} and isinstance(item["content"], str) and bool(item["content"].strip())


def qualify(
    *,
    model: str,
    roles: list[str],
    max_tokens: int,
    routing_mode: str | None,
    behavior_trials: int = 3,
) -> dict[str, Any]:
    if behavior_trials < 1:
        raise ValueError("behavior_trials must be >= 1")
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
            "schema_version": 5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "contract": "portable_plain_text_v1",
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
    reasoning_policy["reasoning_disable_verified"] = not reasoning_requires_live_verification
    reasoning_policy["verification"] = (
        "not_required" if not reasoning_requires_live_verification else "pending_live_request"
    )

    backend = LLMBackend(
        provider="openrouter",
        model=model,
        max_tokens=max_tokens,
        temperature=None,
        top_p=None,
        json_mode=False,
        openrouter_routing_mode=routing_mode,
    )
    results: dict[str, Any] = {}
    for role in roles:
        system, prompt, parser = _probe(role)
        trials: list[dict[str, Any]] = []

        for trial_number in range(1, behavior_trials + 1):
            response: dict[str, str] = {}

            def capture_and_parse(text: str) -> Any:
                response["text"] = text
                return parser(text)

            try:
                parsed, telemetry = call_plain_parsed(
                    backend,
                    system_prompt=system,
                    prompt=prompt,
                    parser=capture_and_parse,
                )
                behavioral_pass = (
                    _generator_passed(parsed) if role.endswith("generator") else True
                )
                if role.endswith("judge"):
                    behavioral_pass = not parsed.accepted and not parsed.assessable
                elif role.endswith("reviewer"):
                    behavioral_pass = not parsed.agreed
                trials.append(
                    {
                        "trial": trial_number,
                        "transport_compatible": True,
                        "contract_pass": True,
                        "behavioral_pass": behavioral_pass,
                        "passed": behavioral_pass,
                        "response": response["text"],
                        "telemetry": telemetry,
                    }
                )
            except PlainOutputContractError as exc:
                trials.append(
                    {
                        "trial": trial_number,
                        "transport_compatible": True,
                        "contract_pass": False,
                        "behavioral_pass": False,
                        "passed": False,
                        "response": exc.response,
                        "telemetry": exc.telemetry,
                        "error": str(exc),
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
        transport_compatible = all(
            trial["transport_compatible"] is True for trial in trials
        )
        contract_pass = all(trial["contract_pass"] is True for trial in trials)
        behavioral_pass = all(trial["behavioral_pass"] is True for trial in trials)
        role_result = {
            "transport_compatible": transport_compatible,
            "contract_pass": contract_pass,
            "behavioral_pass": behavioral_pass,
            "reasoning_policy_pass": True,
            "passed": passed_trials == behavior_trials,
            "behavior_trials_requested": behavior_trials,
            "behavior_trials_passed": passed_trials,
            "trials": trials,
        }
        last_response = next(
            (trial.get("response") for trial in reversed(trials) if trial.get("response")),
            None,
        )
        first_error = next(
            (trial.get("error") for trial in trials if trial.get("error")),
            None,
        )
        if last_response is not None:
            role_result["response"] = last_response
        if first_error is not None:
            role_result["error"] = first_error
        results[role] = role_result
    if reasoning_requires_live_verification:
        reasoning_disable_verified = any(
            result.get("transport_compatible") is True for result in results.values()
        )
        reasoning_policy["reasoning_disable_supported"] = reasoning_disable_verified
        reasoning_policy["reasoning_disable_verified"] = reasoning_disable_verified
        reasoning_policy["reasoning_policy_pass"] = reasoning_disable_verified
        reasoning_policy["verification"] = (
            "live_request_with_reasoning_effort_none"
            if reasoning_disable_verified
            else "live_request_failed"
        )
        for result in results.values():
            result["reasoning_policy_pass"] = reasoning_disable_verified
            result["passed"] = bool(result["passed"] and reasoning_disable_verified)

    return {
        "schema_version": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "contract": "portable_plain_text_v1",
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
    parser.add_argument("--roles", required=True, help="Comma-separated role names or all")
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
        default=None,
    )
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roles = (
        sorted(ROLES)
        if args.roles.strip().lower() == "all"
        else [item.strip() for item in args.roles.split(",") if item.strip()]
    )
    unknown = sorted(set(roles) - ROLES)
    if unknown:
        raise ValueError(f"unknown qualification role(s): {unknown}")
    report = qualify(
        model=args.model,
        roles=roles,
        max_tokens=args.max_tokens,
        routing_mode=args.openrouter_routing_mode,
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

    strict = os.getenv("QUALIFY_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
