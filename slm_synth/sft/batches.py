"""LLM batch formatting and response validation for SFT generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.sft.schema import validate_sft_row
from slm_synth.sft.specs import teacher_visible_sft_spec

SFT_BATCH_RESPONSE_FIELDS = frozenset({"items"})

CHAT_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["user", "assistant"]},
        "content": {"type": "string", "minLength": 1},
    },
}

SFT_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category", "difficulty", "template_family", "eval_family"],
    "properties": {
        "category": {"type": "string", "minLength": 1},
        "difficulty": {"type": "integer"},
        "template_family": {"type": "string", "minLength": 1},
        "eval_family": {"type": ["string", "null"]},
    },
}

SFT_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "messages", "metadata"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "messages": {
                        "type": "array",
                        "items": CHAT_MESSAGE_SCHEMA,
                    },
                    "metadata": SFT_METADATA_SCHEMA,
                },
            },
        }
    },
}


def build_sft_teacher_request_items(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return batchable SFT specs that are safe to send to the LLM."""
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for spec in specs:
        item = teacher_visible_sft_spec(spec)
        item_id = item["id"]
        if item_id in seen_ids:
            duplicate_ids.add(item_id)
        seen_ids.add(item_id)
        items.append(item)

    if duplicate_ids:
        raise ValueError(f"SFT specs contain duplicate id(s): {sorted(duplicate_ids)}")
    return items


def build_sft_teacher_request_object(specs: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the SFT teacher request object."""
    return {"items": build_sft_teacher_request_items(specs)}


def render_sft_batch_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    """Render a JSON-only prompt for batched LLM SFT row generation."""
    request_object = build_sft_teacher_request_object(specs)
    request_json = json.dumps(request_object, ensure_ascii=False, indent=2)

    return (
        "You are generating synthetic supervised fine-tuning data.\n\n"
        "For each input spec, return exactly one SFT training row with the same id.\n"
        "Return only a valid JSON object with this shape:\n"
        '{"items":[{"id":"string","messages":[{"role":"user","content":"string"},'
        '{"role":"assistant","content":"string"}],"metadata":{"category":"string",'
        '"difficulty":1,"template_family":"string","eval_family":"string|null"}}]}\n\n'
        "Rules:\n"
        "- Do not add, remove, rename, or reorder ids.\n"
        "- Preserve metadata values from each input spec exactly.\n"
        "- Generate the final user/assistant training messages from the spec.\n"
        "- Keep answers correct, concise, and aligned with the instruction.\n"
        "- Do not copy known eval prompts exactly.\n"
        "- Do not include variables, constraints, holdout_key, teacher_model, teacher_provider, or generation_run.\n\n"
        "Input specs:\n"
        f"{request_json}"
    )


def validate_sft_batch_response(
    response_object: Mapping[str, Any],
    *,
    expected_ids: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate a batched LLM SFT response and return normalized rows."""
    if not isinstance(response_object, Mapping):
        raise TypeError("SFT batch response must be a mapping")

    keys = set(response_object)
    missing = SFT_BATCH_RESPONSE_FIELDS - keys
    if missing:
        raise ValueError(f"SFT batch response missing required field(s): {sorted(missing)}")

    unexpected = keys - SFT_BATCH_RESPONSE_FIELDS
    if unexpected:
        raise ValueError(f"SFT batch response contains unexpected field(s): {sorted(unexpected)}")

    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("SFT batch response field 'items' must be a list")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"SFT batch response expected {expected_count} item(s), got {len(items)}")

    rows = [validate_sft_row(item) for item in items]
    _validate_response_ids([row["id"] for row in rows], expected_ids=expected_ids)
    return rows


def _validate_response_ids(row_ids: list[str], *, expected_ids: Iterable[str] | None) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row_id in row_ids:
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"SFT batch response contains duplicate id(s): {sorted(duplicates)}")

    if expected_ids is None:
        return

    expected = list(expected_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("expected_ids contains duplicate id(s)")

    actual_set = set(row_ids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing and unexpected:
        raise ValueError(
            f"SFT batch response missing expected id(s): {missing}; "
            f"contains unexpected id(s): {unexpected}"
        )
    if missing:
        raise ValueError(f"SFT batch response missing expected id(s): {missing}")
    if unexpected:
        raise ValueError(f"SFT batch response contains unexpected id(s): {unexpected}")


def validate_sft_rows_against_specs(
    rows: Iterable[Mapping[str, Any]],
    specs: Iterable[Mapping[str, Any]],
) -> None:
    """Validate generated row content against local family specs."""
    spec_by_id = {str(spec["id"]): spec for spec in specs}
    for row in rows:
        row_id = str(row["id"])
        spec = spec_by_id.get(row_id)
        if spec is None:
            raise ValueError(f"SFT row {row_id} has no matching spec")
        _validate_row_metadata_matches_spec(row=row, spec=spec)
        answer = _assistant_answer(row)
        family = str(spec["metadata"].get("eval_family") or "")
        variables = dict(spec.get("variables", {}))
        _validate_family_answer(row_id=row_id, family=family, answer=answer, variables=variables)


def _validate_row_metadata_matches_spec(*, row: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if dict(row["metadata"]) != dict(spec["metadata"]):
        raise ValueError(f"SFT row {row['id']} metadata does not match spec metadata")


def _assistant_answer(row: Mapping[str, Any]) -> str:
    messages = row["messages"]
    if not isinstance(messages, list):
        raise TypeError("SFT row messages must be a list")
    return str(messages[-1]["content"]).strip()


def _validate_family_answer(
    *,
    row_id: str,
    family: str,
    answer: str,
    variables: Mapping[str, Any],
) -> None:
    if family in {
        "basic_arithmetic_qa",
        "code_expression_result",
        "direct_division",
        "direct_subtraction",
        "repeat_exact_n_times",
        "list_exact_n_items",
    }:
        _require_exact_answer(row_id=row_id, answer=answer, expected=variables.get("answer"))
        return

    if family == "capital_city_qa":
        _require_exact_answer(row_id=row_id, answer=answer, expected=variables.get("capital"))
        return

    if family == "clear_sky_color_qa":
        _require_exact_answer(row_id=row_id, answer=answer, expected=variables.get("answer"))
        return

    if family == "short_factual_stop_behavior":
        _require_exact_answer(row_id=row_id, answer=answer, expected=variables.get("capital"))
        return

    if family == "private_or_unverifiable_company_fact":
        _validate_private_company_answer(row_id=row_id, answer=answer)
        return

    if family == "code_generation_function":
        _validate_code_generation_answer(row_id=row_id, answer=answer, variables=variables)
        return

    if family == "function_completion_body_only":
        _validate_function_body_answer(row_id=row_id, answer=answer)
        return

    if family == "code_explanation_no_code":
        _validate_code_explanation_answer(row_id=row_id, answer=answer, variables=variables)
        return

    if family == "ai_concept_explanation":
        _validate_ai_concept_answer(row_id=row_id, answer=answer, variables=variables)
        return


def _require_exact_answer(*, row_id: str, answer: str, expected: Any) -> None:
    if expected is None:
        raise ValueError(f"SFT row {row_id} spec is missing expected answer")
    if answer != str(expected):
        raise ValueError(f"SFT row {row_id} assistant answer must exactly match expected answer")


def _validate_private_company_answer(*, row_id: str, answer: str) -> None:
    lowered = answer.lower()
    restraint_markers = (
        "cannot verify",
        "can't verify",
        "cannot determine",
        "can't determine",
        "do not have access",
        "don't have access",
        "no access",
        "not publicly",
        "not public",
        "not available",
        "unverifiable",
        "private",
        "i do not know",
        "i don't know",
    )
    if not any(marker in lowered for marker in restraint_markers):
        raise ValueError(f"SFT row {row_id} assistant answer must show restraint for private information")
    if any(char.isdigit() for char in answer) or "$" in answer:
        raise ValueError(f"SFT row {row_id} assistant answer must not invent private numeric metrics")


def _validate_code_generation_answer(
    *,
    row_id: str,
    answer: str,
    variables: Mapping[str, Any],
) -> None:
    function_name = variables.get("function_name")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError(f"SFT row {row_id} spec is missing function_name")
    _reject_markdown_fence(row_id=row_id, answer=answer)
    if f"def {function_name}(" not in answer:
        raise ValueError(f"SFT row {row_id} assistant answer must define {function_name}")
    _reject_prose_prefix(row_id=row_id, answer=answer)


def _validate_function_body_answer(*, row_id: str, answer: str) -> None:
    _reject_markdown_fence(row_id=row_id, answer=answer)
    if "def " in answer:
        raise ValueError(f"SFT row {row_id} assistant answer must not repeat a function signature")
    _reject_prose_prefix(row_id=row_id, answer=answer)


def _validate_code_explanation_answer(
    *,
    row_id: str,
    answer: str,
    variables: Mapping[str, Any],
) -> None:
    _reject_markdown_fence(row_id=row_id, answer=answer)
    expected_result = variables.get("expected_result")
    if expected_result is not None and not _contains_expected_result(answer=answer, expected=expected_result):
        raise ValueError(f"SFT row {row_id} assistant explanation must mention expected result")
    snippet = variables.get("snippet")
    if isinstance(snippet, str) and snippet.strip() and snippet.strip() in answer:
        raise ValueError(f"SFT row {row_id} assistant explanation must not reproduce the full code")


def _validate_ai_concept_answer(
    *,
    row_id: str,
    answer: str,
    variables: Mapping[str, Any],
) -> None:
    _reject_markdown_fence(row_id=row_id, answer=answer)
    expected_content = variables.get("expected_content")
    if isinstance(expected_content, str) and expected_content.strip():
        expected_terms = _important_terms(expected_content)
        answer_lowered = answer.lower()
        if expected_terms and not any(term in answer_lowered for term in expected_terms):
            raise ValueError(f"SFT row {row_id} assistant explanation must cover expected concept content")
    else:
        concept = variables.get("concept")
        if isinstance(concept, str) and concept.lower() not in answer.lower():
            raise ValueError(f"SFT row {row_id} assistant explanation must mention the target concept")
    if len(answer.split()) > 80:
        raise ValueError(f"SFT row {row_id} assistant explanation is too long")


def _contains_expected_result(*, answer: str, expected: Any) -> bool:
    expected_text = str(expected)
    if expected_text in answer:
        return True
    return expected_text.replace(" ", "") in answer.replace(" ", "")


def _important_terms(text: str) -> tuple[str, ...]:
    stopwords = {
        "that",
        "with",
        "from",
        "into",
        "used",
        "uses",
        "items",
        "model",
    }
    terms: list[str] = []
    for raw in text.lower().replace("-", " ").split():
        term = "".join(char for char in raw if char.isalnum())
        if len(term) >= 5 and term not in stopwords:
            terms.append(term)
    return tuple(terms)


def _reject_markdown_fence(*, row_id: str, answer: str) -> None:
    if "```" in answer:
        raise ValueError(f"SFT row {row_id} assistant answer must not use Markdown fences")


def _reject_prose_prefix(*, row_id: str, answer: str) -> None:
    first_line = answer.lstrip().splitlines()[0].lower()
    prose_prefixes = ("here", "sure", "this", "the ")
    if first_line.startswith(prose_prefixes):
        raise ValueError(f"SFT row {row_id} assistant answer must not include prose outside code")
