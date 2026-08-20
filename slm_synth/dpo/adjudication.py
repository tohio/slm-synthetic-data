"""Independent semantic quality and preference-separation gate for DPO."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec

DPO_ADJUDICATION_SCORES = (
    "chosen_quality",
    "rejected_plausibility",
    "weakness_match",
    "preference_separation",
    "collateral_preservation",
)

DPO_ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": [
            "id", "accepted", "preference_dimension", "failure_mode",
            "observed_weakness", "scores", "constraint_results", "reasons",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "accepted": {"type": "boolean"},
            "preference_dimension": {"type": "string", "minLength": 1},
            "failure_mode": {"type": "string", "minLength": 1},
            "observed_weakness": {"type": "string", "minLength": 1},
            "scores": {
                "type": "object", "additionalProperties": False,
                "required": list(DPO_ADJUDICATION_SCORES),
                "properties": {
                    name: {"type": "integer", "minimum": 1, "maximum": 4}
                    for name in DPO_ADJUDICATION_SCORES
                },
            },
            "constraint_results": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["constraint", "passed", "reason"],
                "properties": {
                    "constraint": {"type": "string", "minLength": 1},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                },
            }},
            "reasons": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
    }}},
}


class StructuredAdjudicatorBackend(Protocol):
    def generate_structured_object_with_metadata(
        self, *, prompt: str, schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any]: ...


def adjudicate_dpo_rows(
    *, specs: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]],
    backend: StructuredAdjudicatorBackend
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    payload = {"items": [
        {"spec": teacher_visible_dpo_spec(spec), "rendered_pair": dict(row)}
        for spec, row in zip(validated_specs, rows, strict=True)
    ]}
    prompt = (
        "Independently adjudicate these DPO candidates. Return only JSON matching the schema. The chosen response must be "
        "high quality. The rejected response must be plausible and contain exactly the requested failure_mode on the named "
        "preference_dimension while preserving unrelated strengths. Reject arbitrary corruption, multiple weaknesses, copied "
        "branches, unsupported facts, or a numeric error not explicitly grounded in the brief. Score from 1 to 4 and accept "
        "only when every score is at least 3 and every source constraint passes. Do not repair any row.\n\n"
        "Copy each source constraint into constraint_results exactly and in the original order.\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    result = backend.generate_structured_object_with_metadata(
        prompt=prompt, schema=DPO_ADJUDICATION_SCHEMA, schema_name="dpo_quality_adjudication"
    )
    data = result.get("data")
    if not isinstance(data, Mapping) or set(data) != {"items"} or not isinstance(data["items"], list):
        raise ValueError("DPO adjudicator returned invalid data")
    expected_ids = [spec["id"] for spec in validated_specs]
    specs_by_id = {spec["id"]: spec for spec in validated_specs}
    decisions: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for raw in data["items"]:
        if not isinstance(raw, Mapping):
            raise TypeError("DPO adjudication item must be an object")
        item = dict(raw)
        item_id = str(item.get("id"))
        if item_id in decisions:
            raise ValueError(f"DPO adjudication contains duplicate id: {item_id}")
        spec = specs_by_id.get(item_id)
        if spec is None:
            raise ValueError(f"DPO adjudication contains unexpected id: {item_id}")
        metadata = spec["metadata"]
        if item.get("preference_dimension") != metadata["preference_dimension"]:
            failures.append(f"{item_id}: preference_dimension mismatch")
        if item.get("failure_mode") != metadata["failure_mode"]:
            failures.append(f"{item_id}: failure_mode mismatch")
        scores = item.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(DPO_ADJUDICATION_SCORES):
            raise ValueError(f"DPO adjudication scores are invalid for {item_id}")
        scores_pass = all(
            isinstance(scores[name], int) and not isinstance(scores[name], bool) and 3 <= scores[name] <= 4
            for name in DPO_ADJUDICATION_SCORES
        )
        results = item.get("constraint_results")
        constraints = list(spec.get("constraints", []))
        if not isinstance(results, list) or [entry.get("constraint") for entry in results if isinstance(entry, Mapping)] != constraints:
            failures.append(f"{item_id}: constraint adjudication does not match the source brief")
        elif not all(entry.get("passed") is True for entry in results):
            failures.append(f"{item_id}: source constraint failed")
        if item.get("accepted") is not True or not scores_pass:
            reasons = item.get("reasons")
            detail = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else "quality gate failed"
            failures.append(f"{item_id}: {detail or 'quality gate failed'}")
        decisions[item_id] = item
    if set(decisions) != set(expected_ids):
        raise ValueError(f"DPO adjudication id mismatch: missing={sorted(set(expected_ids) - set(decisions))}")
    if failures:
        raise ValueError("semantic DPO adjudication rejected candidate(s): " + " | ".join(failures))
    telemetry = result.get("telemetry")
    return decisions, dict(telemetry) if isinstance(telemetry, Mapping) else {}
