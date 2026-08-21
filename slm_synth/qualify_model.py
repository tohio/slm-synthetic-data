"""Qualify an OpenRouter model against portable repository roles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.llm import LLMBackend
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
    *, model: str, roles: list[str], max_tokens: int, routing_mode: str | None
) -> dict[str, Any]:
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
            behavioral_pass = _generator_passed(parsed) if role.endswith("generator") else True
            if role.endswith("judge"):
                behavioral_pass = not parsed.accepted and not parsed.assessable
            elif role.endswith("reviewer"):
                behavioral_pass = not parsed.agreed
            results[role] = {
                "transport_compatible": True,
                "contract_pass": True,
                "behavioral_pass": behavioral_pass,
                "passed": behavioral_pass,
                "response": response["text"],
                "telemetry": telemetry,
            }
        except PlainOutputContractError as exc:
            results[role] = {
                "transport_compatible": True,
                "contract_pass": False,
                "behavioral_pass": False,
                "passed": False,
                "response": exc.response,
                "telemetry": exc.telemetry,
                "error": str(exc),
            }
        except Exception as exc:
            results[role] = {
                "transport_compatible": False,
                "contract_pass": False,
                "behavioral_pass": False,
                "passed": False,
                "error": str(exc),
            }
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "contract": "portable_plain_text_v1",
        "roles": results,
        "passed": all(result["passed"] for result in results.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--roles", required=True, help="Comma-separated role names or all")
    parser.add_argument("--max-tokens", type=int, default=512)
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
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote model qualification report to {path}")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
