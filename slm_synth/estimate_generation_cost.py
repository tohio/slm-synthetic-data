"""Estimate generator/judge/reviewer OpenRouter cost before a run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.request import urlopen


def _pricing(models: list[str], pricing_file: str | None) -> dict[str, dict[str, float]]:
    if pricing_file:
        payload = json.loads(Path(pricing_file).read_text(encoding="utf-8"))
    else:
        with urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:  # nosec B310
            payload = json.load(response)
    records = payload.get("data", payload)
    by_id = {item["id"]: item for item in records}
    result: dict[str, dict[str, float]] = {}
    for model in models:
        if model not in by_id:
            raise ValueError(f"OpenRouter pricing not found for model {model!r}")
        price = by_id[model].get("pricing", {})
        result[model] = {
            "prompt": float(price.get("prompt", 0)),
            "completion": float(price.get("completion", 0)),
        }
    return result


def estimate(args: argparse.Namespace) -> dict:
    for name in (
        "deterministic_pass_rate",
        "judge_acceptance",
        "reviewer_agreement",
    ):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than 0 and at most 1")
    if args.average_accepted_tokens < 1:
        raise ValueError("--average-accepted-tokens must be positive")
    rates = _pricing([args.generator_model, args.judge_model, args.reviewer_model], args.pricing_file)
    target_accepted = args.target_accepted
    if args.target_tokens is not None:
        target_accepted = math.ceil(args.target_tokens / args.average_accepted_tokens)
    scenarios = {
        "low": (
            min(0.98, args.deterministic_pass_rate + 0.05),
            min(0.95, args.judge_acceptance + 0.10),
            min(0.97, args.reviewer_agreement + 0.07),
            0.80,
            1.00,
        ),
        "expected": (
            args.deterministic_pass_rate,
            args.judge_acceptance,
            args.reviewer_agreement,
            1.00,
            1.00,
        ),
        "high": (
            max(0.40, args.deterministic_pass_rate - 0.15),
            max(0.40, args.judge_acceptance - 0.15),
            max(0.40, args.reviewer_agreement - 0.15),
            1.25,
            1.20,
        ),
    }
    output = {}
    for name, (
        deterministic_pass_rate,
        judge_acceptance,
        reviewer_agreement,
        token_multiplier,
        retry_multiplier,
    ) in scenarios.items():
        candidates = args.candidates
        if target_accepted is not None:
            candidates = math.ceil(
                target_accepted
                / (deterministic_pass_rate * judge_acceptance * reviewer_agreement)
            )
        if candidates is None:
            raise ValueError("provide --candidates, --target-accepted, or --target-tokens")
        judge_calls = candidates * deterministic_pass_rate
        review_calls = judge_calls * judge_acceptance
        accepted = max(1.0, review_calls * reviewer_agreement)
        calls = {
            "generator": (args.generator_model, candidates, args.generator_input_tokens, args.generator_output_tokens),
            "judge": (args.judge_model, judge_calls, args.judge_input_tokens, args.judge_output_tokens),
            "reviewer": (args.reviewer_model, review_calls, args.reviewer_input_tokens, args.reviewer_output_tokens),
        }
        role_costs = {
            role: count * retry_multiplier * token_multiplier * (
                input_tokens * rates[model]["prompt"] + output_tokens * rates[model]["completion"]
            )
            for role, (model, count, input_tokens, output_tokens) in calls.items()
        }
        total = sum(role_costs.values())
        output[name] = {
            "candidate_calls": candidates,
            "deterministic_pass_rate": deterministic_pass_rate,
            "judge_calls": round(judge_calls, 2),
            "reviewer_calls": round(review_calls, 2),
            "estimated_accepted": round(accepted, 2),
            "estimated_accepted_tokens": round(accepted * args.average_accepted_tokens),
            "role_costs_usd": role_costs,
            "total_cost_usd": total,
            "cost_per_accepted_usd": total / accepted,
        }
    return {"pricing_source": "OpenRouter", "models": rates, "scenarios": output}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator-model", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--reviewer-model", required=True)
    p.add_argument("--candidates", type=int)
    p.add_argument("--target-accepted", type=int)
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--average-accepted-tokens", type=int, default=500)
    p.add_argument("--pricing-file")
    p.add_argument("--deterministic-pass-rate", type=float, default=0.90)
    p.add_argument("--judge-acceptance", type=float, default=0.75)
    p.add_argument("--reviewer-agreement", type=float, default=0.85)
    p.add_argument("--generator-input-tokens", type=int, default=1200)
    p.add_argument("--generator-output-tokens", type=int, default=500)
    p.add_argument("--judge-input-tokens", type=int, default=1700)
    p.add_argument("--judge-output-tokens", type=int, default=80)
    p.add_argument("--reviewer-input-tokens", type=int, default=1800)
    p.add_argument("--reviewer-output-tokens", type=int, default=50)
    p.add_argument("--output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = estimate(args)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
