from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional


_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def attach_candidate_ids(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach local positional ids for matching response items to candidates."""
    return [{"candidate_id": index, **candidate} for index, candidate in enumerate(candidates)]


def order_responses(responses: Iterable[Dict[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    """Return response records in candidate order or fail the request for retry/split."""
    indexed: Dict[int, Dict[str, Any]] = {}
    for response in responses:
        candidate_id = response.get("candidate_id")
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise ValueError("Response item is missing integer candidate_id")
        if not 0 <= candidate_id < expected_count:
            raise ValueError(f"Response candidate_id out of range: {candidate_id}")
        if candidate_id in indexed:
            raise ValueError(f"Duplicate response candidate_id: {candidate_id}")
        indexed[candidate_id] = response

    missing = [index for index in range(expected_count) if index not in indexed]
    if missing:
        raise ValueError(f"Missing response candidate_id values: {missing}")
    return [indexed[index] for index in range(expected_count)]


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _integer_text(value: Any) -> Optional[str]:
    text = _text(value).replace(",", "")
    if not _INTEGER_RE.fullmatch(text):
        return None
    return str(int(text))


def _choice_position(question: str, answer: str) -> int:
    """Select a deterministic answer slot without delegating answer indexing to the LLM."""
    digest = hashlib.sha256(f"{question}|{answer}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 4


def finalize_math_mcq(candidate: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble math MCQ bookkeeping locally from the response model's solved answer."""
    question = _text(candidate.get("question"))
    answer = _integer_text(response.get("verification_answer"))
    original_choices = candidate.get("choices") if isinstance(candidate.get("choices"), list) else []

    # Invalid answer metadata is deliberately left for record validation to reject.
    if answer is None:
        return {
            "type": "educational_qa_mcq_math",
            "question": question,
            "choices": [_text(choice) for choice in original_choices],
            "correct_index": None,
            "explanation": _text(response.get("explanation")),
            "verification_expression": _text(response.get("verification_expression")),
            "verification_answer": _text(response.get("verification_answer")),
        }

    distractors: List[str] = []
    for choice in original_choices:
        normalized = _integer_text(choice)
        if normalized is not None and normalized != answer and normalized not in distractors:
            distractors.append(normalized)
        if len(distractors) == 3:
            break

    answer_value = int(answer)
    offsets = [1, -1, 2, -2, 5, -5, 10, -10, 20, -20]
    for offset in offsets:
        proposed = str(answer_value + offset)
        if proposed != answer and proposed not in distractors:
            distractors.append(proposed)
        if len(distractors) == 3:
            break

    choices = distractors[:3]
    position = _choice_position(question, answer)
    choices.insert(position, answer)
    return {
        "type": "educational_qa_mcq_math",
        "question": question,
        "choices": choices,
        "correct_index": position,
        "explanation": _text(response.get("explanation")),
        "verification_expression": _text(response.get("verification_expression")),
        "verification_answer": answer,
    }


def finalize_general_mcq(candidate: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the general-MCQ answer index from answer text instead of model-supplied indexing."""
    question = _text(candidate.get("question"))
    answer = _text(response.get("answer"))
    supplied = candidate.get("choices") if isinstance(candidate.get("choices"), list) else []

    choices: List[str] = []
    for choice in supplied:
        value = _text(choice)
        if value and value.casefold() not in {existing.casefold() for existing in choices}:
            choices.append(value)
        if len(choices) == 4:
            break

    index = next(
        (position for position, choice in enumerate(choices) if answer and choice.casefold() == answer.casefold()),
        None,
    )
    if index is None and answer:
        if len(choices) >= 4:
            index = _choice_position(question, answer)
            choices[index] = answer
        else:
            choices.append(answer)
            index = len(choices) - 1

    return {
        "type": "educational_qa_mcq_general",
        "question": question,
        "choices": choices,
        "correct_index": index,
        "explanation": _text(response.get("explanation")),
    }
