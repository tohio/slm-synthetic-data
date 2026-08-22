"""Portable judge and reviewer gates for generated SFT candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.model_contract import PlainTextBackend, call_plain_parsed, parse_judge_decision, parse_review_decision
from slm_synth.quality_telemetry import combine_telemetry
from slm_synth.sft.specs import validate_sft_spec


def _public_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only evidence the semantic quality roles are allowed to judge.

    The public conversation is authoritative for task requirements. Repository-owned
    planning metadata, hidden variables, taxonomy labels, and exact output constraints
    are deliberately excluded. Exact/structural constraints have already passed local
    deterministic validation before semantic adjudication begins.
    """
    return {
        "public_conversation": list(row.get("messages", [])),
        "deterministic_validation": {
            "status": "passed",
            "instruction": (
                "Repository-owned exact and structural checks passed before semantic review. "
                "Treat that result as supporting evidence while still reviewing the candidate holistically."
            ),
        },
    }


def _judge_prompt(row: Mapping[str, Any]) -> str:
    payload = _public_evidence(row)
    return (
        "Decide whether this candidate is suitable training data from the public conversation and candidate content. "
        "Evaluate correctness, grounding, completeness, instruction adherence, and material ambiguity. Reject only when "
        "there is a concrete, material defect supported by the supplied evidence. Do not reject merely because another "
        "interpretation is possible, wording could be improved, an unstated detail could be more explicit, or you would "
        "personally answer differently. Do not treat hidden repository metadata, variables, taxonomy labels, or planning "
        "fields as independent requirements. Deterministic checks are supplied as supporting evidence; review the whole "
        "candidate, but do not invent conflicting facts about those checks. Never guess and never repair the candidate. "
        "Do not invent stronger coverage requirements than the user explicitly requested; a valid demonstration of a "
        "required condition is sufficient unless exhaustive coverage is explicitly requested. When rejecting, identify the "
        "specific candidate content and public requirement or evidence that makes the defect material.\n\n"
        "Return exactly three labeled lines:\nASSESSABLE: YES or NO\nDECISION: ACCEPT or REJECT\n"
        "REASON: one concise evidence-based reason\n\n"
        f"Candidate evidence:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _review_prompt(row: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    payload = {**_public_evidence(row), "judge": dict(judge)}
    return (
        "Review only whether the judge's ACCEPT decision is reasonably justified by the public conversation and "
        "candidate. AGREE unless there is a clear, material defect the judge missed. Do not perform a fresh stricter "
        "re-judgment, search for a novel defect merely to overturn the judge, or reject for stylistic preferences, harmless "
        "ambiguity, or an alternative but reasonable interpretation. Do not invent requirements from hidden repository "
        "metadata, variables, taxonomy labels, or planning fields. Deterministic checks are supporting evidence; review the "
        "whole candidate without inventing conflicting facts about those checks. If you disagree, identify the specific "
        "candidate content and public requirement or evidence that makes the judge's acceptance materially wrong. Do not "
        "repair the candidate or include self-deliberation. Your AGREE label must match your reason: if your reason says "
        "the judge's acceptance is reasonably justified, output AGREE: YES; output AGREE: NO only when your reason identifies "
        "a clear material defect the judge missed.\n\n"
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
        judged, call_telemetry = call_plain_parsed(
            backend, system_prompt="You are an evidence-based dataset quality judge. Use only supplied evidence.",
            prompt=_judge_prompt(row),
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
                reviewer_backend, system_prompt="You independently audit whether dataset acceptance decisions are reasonably justified.",
                prompt=_review_prompt(row, decision),
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
