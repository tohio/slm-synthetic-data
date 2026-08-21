"""Portable judge and reviewer gates for generated SFT candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.model_contract import PlainTextBackend, call_plain_parsed, parse_judge_decision, parse_review_decision
from slm_synth.quality_telemetry import combine_telemetry
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec


def _judge_prompt(spec: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    payload = {"brief": dict(spec), "candidate": dict(row)}
    return (
        "Decide whether this candidate is unambiguous, answerable from its public conversation, correct, grounded, "
        "complete, and compliant with every instruction and constraint. Reject it if the task or evidence is ambiguous, "
        "insufficient, contradictory, or cannot be evaluated reliably. Never guess and never repair the candidate.\n\n"
        "Return exactly three labeled lines:\nASSESSABLE: YES or NO\nDECISION: ACCEPT or REJECT\n"
        "REASON: one concise evidence-based reason\n\n"
        f"Candidate:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _review_prompt(spec: Mapping[str, Any], row: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    payload = {"brief": dict(spec), "candidate": dict(row), "judge": dict(judge)}
    return (
        "Review only whether the judge's ACCEPT decision is justified by the supplied brief and candidate. "
        "Disagree if the judge missed ambiguity, missing evidence, an incorrect claim, or any unmet constraint. "
        "Do not repair the candidate and do not introduce new requirements.\n\n"
        "Return exactly two labeled lines:\nAGREE: YES or NO\nREASON: one concise evidence-based reason\n\n"
        f"Review item:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def adjudicate_sft_rows(
    *, specs: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]],
    backend: PlainTextBackend, reviewer_backend: PlainTextBackend,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Judge candidates, then review only judge-accepted candidates."""
    validated_specs = [validate_sft_spec(spec) for spec in specs]
    rendered_rows = [dict(row) for row in rows]
    decisions: dict[str, dict[str, Any]] = {}
    judge_telemetry: list[dict[str, Any]] = []
    reviewer_telemetry: list[dict[str, Any]] = []
    for spec, row in zip(validated_specs, rendered_rows, strict=True):
        visible_spec = teacher_visible_sft_spec(spec)
        judged, call_telemetry = call_plain_parsed(
            backend, system_prompt="You are a conservative dataset quality judge. Use only supplied evidence.",
            prompt=_judge_prompt(visible_spec, row),
            parser=parse_judge_decision,
        )
        judge_telemetry.append(call_telemetry)
        decision: dict[str, Any] = {
            "id": spec["id"], "assessable": judged.assessable,
            "judge_accepted": judged.accepted, "judge_reason": judged.reason,
            "reviewed": False, "reviewer_agreed": False, "accepted": False,
        }
        if judged.accepted:
            reviewed, review_telemetry = call_plain_parsed(
                reviewer_backend, system_prompt="You independently audit dataset acceptance decisions.",
                prompt=_review_prompt(visible_spec, row, decision),
                parser=parse_review_decision,
            )
            reviewer_telemetry.append(review_telemetry)
            decision.update(
                reviewed=True, reviewer_agreed=reviewed.agreed,
                reviewer_reason=reviewed.reason, accepted=reviewed.agreed,
            )
        decisions[spec["id"]] = decision
    aggregate = combine_telemetry(*judge_telemetry, *reviewer_telemetry)
    aggregate["role_telemetry"] = {
        "judge": combine_telemetry(*judge_telemetry),
        "reviewer": combine_telemetry(*reviewer_telemetry),
    }
    return decisions, aggregate
