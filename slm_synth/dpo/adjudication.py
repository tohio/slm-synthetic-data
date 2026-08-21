"""Portable judge and reviewer gates for DPO preference pairs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec
from slm_synth.model_contract import PlainTextBackend, call_plain_parsed, parse_judge_decision, parse_review_decision
from slm_synth.quality_telemetry import combine_telemetry


def _judge_prompt(spec: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    return (
        "Decide whether this DPO pair is reliably assessable and suitable. The chosen branch must be correct, grounded, "
        "complete, and clearly better on the named preference dimension. The rejected branch must remain plausible while "
        "showing exactly the requested controlled failure, without unrelated corruption. Reject ambiguity, insufficient "
        "evidence, copied branches, arbitrary wrong numbers, multiple weaknesses, or any case you cannot determine. Never "
        "guess or repair the pair.\n\nReturn exactly three labeled lines:\nASSESSABLE: YES or NO\n"
        "DECISION: ACCEPT or REJECT\nREASON: one concise evidence-based reason\n\n"
        f"Pair:\n{json.dumps({'brief': dict(spec), 'candidate': dict(row)}, ensure_ascii=False, indent=2)}"
    )


def _review_prompt(spec: Mapping[str, Any], row: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    return (
        "Review only whether the judge's ACCEPT decision is justified. Disagree if chosen quality, rejected plausibility, "
        "the requested weakness, preference separation, grounding, or assessability was judged incorrectly. Do not repair "
        "the pair or add requirements.\n\nReturn exactly two labeled lines:\nAGREE: YES or NO\n"
        "REASON: one concise evidence-based reason\n\n"
        f"Review item:\n{json.dumps({'brief': dict(spec), 'candidate': dict(row), 'judge': dict(judge)}, ensure_ascii=False, indent=2)}"
    )


def adjudicate_dpo_rows(
    *, specs: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]],
    backend: PlainTextBackend, reviewer_backend: PlainTextBackend,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    rendered_rows = [dict(row) for row in rows]
    decisions: dict[str, dict[str, Any]] = {}
    judge_telemetry: list[dict[str, Any]] = []
    reviewer_telemetry: list[dict[str, Any]] = []
    for spec, row in zip(validated_specs, rendered_rows, strict=True):
        visible_spec = teacher_visible_dpo_spec(spec)
        judged, call_telemetry = call_plain_parsed(
            backend, system_prompt="You are a conservative preference-data quality judge. Use only supplied evidence.",
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
                reviewer_backend, system_prompt="You independently audit preference-data acceptance decisions.",
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
