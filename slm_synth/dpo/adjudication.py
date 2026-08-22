"""Portable judge and reviewer gates for DPO preference pairs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.dpo.specs import validate_dpo_spec
from slm_synth.model_contract import PlainTextBackend, call_plain_parsed, parse_judge_decision, parse_review_decision
from slm_synth.quality_telemetry import combine_telemetry


def _public_evidence(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Return public pair evidence plus only the intended preference dimension.

    Hidden variables, taxonomy/planning fields, and repository-owned exact constraints
    are excluded. The preference dimension is retained because it defines what the
    chosen/rejected comparison is intended to isolate. Exact/structural constraints
    already passed local deterministic validation before semantic adjudication.
    """
    metadata = spec.get("metadata", {})
    preference_dimension = None
    if isinstance(metadata, Mapping):
        preference_dimension = metadata.get("preference_dimension")
    payload: dict[str, Any] = {
        "candidate": dict(row),
        "deterministic_validation": {
            "status": "passed",
            "instruction": (
                "Repository-owned exact and structural checks passed before semantic review. "
                "Treat that result as supporting evidence while still reviewing the pair holistically."
            ),
        },
    }
    if preference_dimension:
        payload["preference_dimension"] = preference_dimension
    return payload


def _judge_prompt(spec: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    payload = _public_evidence(spec, row)
    return (
        "Decide whether this DPO pair is suitable preference-training data from the public pair evidence. The chosen branch "
        "must be correct, grounded, complete, and materially better on the supplied preference dimension. The rejected "
        "branch must remain plausible while exhibiting the intended controlled weakness without unrelated corruption. "
        "Reject only when there is a concrete, material defect supported by the supplied evidence. Do not reject merely "
        "because another interpretation is possible, wording could be improved, or you would personally prefer a different "
        "answer. Do not invent requirements from hidden repository variables, taxonomy labels, or planning fields. "
        "Deterministic checks are supporting evidence; review the whole pair, but do not invent conflicting facts about "
        "those checks. Do not invent stronger coverage requirements than the public task explicitly requests. Never guess "
        "or repair the pair. When rejecting, identify the specific branch content and preference requirement or evidence "
        "that makes the defect material.\n\nReturn exactly three labeled lines:\nASSESSABLE: YES or NO\n"
        "DECISION: ACCEPT or REJECT\nREASON: one concise evidence-based reason\n\n"
        f"Pair evidence:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _review_prompt(spec: Mapping[str, Any], row: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    payload = {**_public_evidence(spec, row), "judge": dict(judge)}
    return (
        "Review only whether the judge's ACCEPT decision is reasonably justified by the public pair evidence and supplied "
        "preference dimension. AGREE unless there is a clear, material defect the judge missed. Do not perform a fresh "
        "stricter re-judgment, search for a novel defect merely to overturn the judge, or reject for stylistic preferences, "
        "harmless ambiguity, or an alternative but reasonable interpretation. Do not invent requirements from hidden "
        "repository variables, taxonomy labels, or planning fields. Deterministic checks are supporting evidence; review the "
        "whole pair without inventing conflicting facts about those checks. If you disagree, identify the specific branch "
        "content and preference requirement or evidence that makes the judge's acceptance materially wrong. Do not repair "
        "the pair or include self-deliberation. Your AGREE label must match your reason: if your reason says the judge's "
        "acceptance is reasonably justified, output AGREE: YES; output AGREE: NO only when your reason identifies a clear "
        "material defect the judge missed.\n\nReturn exactly two labeled lines:\nAGREE: YES or NO\n"
        "REASON: one concise evidence-based reason\n\n"
        f"Review item:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
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
        judged, call_telemetry = call_plain_parsed(
            backend, system_prompt="You are an evidence-based preference-data quality judge. Use only supplied evidence.",
            prompt=_judge_prompt(spec, row),
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
                reviewer_backend, system_prompt="You independently audit whether preference-data acceptance decisions are reasonably justified.",
                prompt=_review_prompt(spec, row, decision),
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
