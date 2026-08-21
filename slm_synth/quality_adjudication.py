"""Strict semantic-adjudication contracts for generated alignment data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

QUALITY_SCORE_NAMES = (
    "correctness",
    "grounding",
    "instruction_adherence",
    "completeness",
    "coherence",
)

QUALITY_ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "accepted", "scores", "constraint_results", "reasons"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "accepted": {"type": "boolean"},
                    "scores": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(QUALITY_SCORE_NAMES),
                        "properties": {
                            name: {"type": "integer", "minimum": 1, "maximum": 4}
                            for name in QUALITY_SCORE_NAMES
                        },
                    },
                    "constraint_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["constraint_index", "passed", "reason"],
                            "properties": {
                                "constraint_index": {"type": "integer", "minimum": 0},
                                "passed": {"type": "boolean"},
                                "reason": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}


def render_quality_adjudication_prompt(
    *, dataset_type: str, specs: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]]
) -> str:
    payload = {
        "items": [
            {"spec": dict(spec), "rendered_row": dict(row)}
            for spec, row in zip(specs, rows, strict=True)
        ]
    }
    return (
        f"Independently adjudicate these rendered {dataset_type} candidates. Return only JSON matching the schema. "
        "Use the complete grounded brief: instruction, variables, constraints, interaction/output/context metadata, "
        "and the rendered row. First verify that every source value needed to understand and answer the task appears in the "
        "public conversation; reject any row whose assistant relies on hidden spec variables. For supplied_passage, "
        "multi_document, and long_document tasks, grounding permits ordinary linguistic entailment and direct inference but "
        "not unsupported factual claims. For self_contained creative, conversational, planning, and brainstorming tasks, "
        "appropriate invented details are allowed unless an explicit requirement prohibits them. Planning and recommendation "
        "responses may propose obtaining a resource, but must not assert that staff, volunteers, partners, supplies, access, "
        "or other resources already exist unless the brief supplies them; assumptions must be labeled. Judge creative content "
        "for consistency with the brief rather than demanding that every fictional detail appear in variables. Treat "
        "meaning-preserving edits as preserving uncertainty and limitations even when wording changes. For classification, "
        "reject a candidate if a supplied label set or decision rule is missing from the public conversation or is not applied. "
        "For extraction and summarization, compare the response against every source statement, entity, field, owner, date, "
        "dependency, limitation, and unresolved item; any requested omission fails completeness. Treat predictive assurances "
        "as promises when the brief prohibits promises about outcomes. Enforce explicit word counts, item counts, headings, "
        "formatting, and forbidden terms. A claimed independent verification must use a genuinely distinct check, not repeat "
        "the same calculation. A candidate passes only when it is correct, grounded, instruction-following, complete, "
        "coherent, and satisfies every stated constraint. Score each criterion from 1 (failed) to 4 (excellent). "
        "Set accepted=true only when every score is at least 3 and every constraint passes. Do not repair or rewrite rows.\n\n"
        "Evaluate every source constraint independently before assigning scores. Each constraint_results reason must identify "
        "specific evidence from the public conversation and response, or identify the exact missing or violating content; a "
        "generic statement such as 'satisfied' is not evidence. Return one constraint_results entry for every source constraint. "
        "Identify each constraint by its zero-based "
        "position in the source constraints array: return constraint_index values 0 through N-1 exactly once and in "
        "ascending order. Do not copy or paraphrase the constraint text into constraint_results.\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def validate_quality_adjudication(
    response: Mapping[str, Any], *, specs: Iterable[Mapping[str, Any]], minimum_score: int = 3
) -> dict[str, dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != {"items"}:
        raise ValueError("quality adjudication must contain only an items field")
    items = response["items"]
    if not isinstance(items, list):
        raise TypeError("quality adjudication items must be a list")
    validated_specs = [dict(spec) for spec in specs]
    expected = [str(spec["id"]) for spec in validated_specs]
    if len(items) != len(expected):
        raise ValueError(f"quality adjudication expected {len(expected)} item(s), got {len(items)}")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            raise TypeError("quality adjudication item must be an object")
        required = {"id", "accepted", "scores", "constraint_results", "reasons"}
        if set(raw) != required:
            raise ValueError("quality adjudication item fields do not match the contract")
        item_id = str(raw["id"])
        if item_id in by_id:
            raise ValueError(f"quality adjudication contains duplicate id: {item_id}")
        scores = raw["scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(QUALITY_SCORE_NAMES):
            raise ValueError(f"quality adjudication scores are invalid for {item_id}")
        normalized_scores: dict[str, int] = {}
        for name in QUALITY_SCORE_NAMES:
            score = scores[name]
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise ValueError(f"quality adjudication score {name} is invalid for {item_id}")
            normalized_scores[name] = score
        constraint_results = raw["constraint_results"]
        reasons = raw["reasons"]
        if not isinstance(constraint_results, list) or not isinstance(reasons, list):
            raise TypeError(f"quality adjudication lists are invalid for {item_id}")
        normalized_results: list[dict[str, Any]] = []
        for result in constraint_results:
            if not isinstance(result, Mapping):
                raise TypeError(f"quality adjudication constraint result is invalid for {item_id}")
            if set(result) != {"constraint_index", "passed", "reason"}:
                raise ValueError(f"quality adjudication constraint result fields are invalid for {item_id}")
            constraint_index = result["constraint_index"]
            if not isinstance(constraint_index, int) or isinstance(constraint_index, bool) or constraint_index < 0:
                raise ValueError(f"quality adjudication constraint index is invalid for {item_id}")
            normalized_results.append(dict(result))
        by_id[item_id] = {
            "id": item_id,
            "accepted": raw["accepted"] is True,
            "scores": normalized_scores,
            "constraint_results": normalized_results,
            "reasons": list(reasons),
        }

    if set(by_id) != set(expected):
        raise ValueError(
            "quality adjudication id mismatch: "
            f"missing={sorted(set(expected) - set(by_id))}, unexpected={sorted(set(by_id) - set(expected))}"
        )
    specs_by_id = {str(spec["id"]): spec for spec in validated_specs}
    failures: list[str] = []
    for item_id in expected:
        item = by_id[item_id]
        constraints = list(specs_by_id[item_id].get("constraints", []))
        results = item["constraint_results"]
        result_indexes = [result["constraint_index"] for result in results]
        if result_indexes != list(range(len(constraints))):
            failures.append(f"{item_id}: constraint index coverage does not match the source brief")
            continue
        passed = all(result.get("passed") is True for result in results)
        scores_pass = all(score >= minimum_score for score in item["scores"].values())
        if not item["accepted"] or not passed or not scores_pass:
            reason = "; ".join(str(value) for value in item["reasons"]) or "semantic quality gate failed"
            failures.append(f"{item_id}: {reason}")
    if failures:
        raise ValueError("semantic quality adjudication rejected candidate(s): " + " | ".join(failures))
    return by_id


def combine_telemetry(*items: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate several provider calls while retaining role-level telemetry."""
    from slm_synth.telemetry import aggregate_llm_telemetry

    return aggregate_llm_telemetry([dict(item) for item in items if item])
