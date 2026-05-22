from __future__ import annotations

from typing import Any, Dict, Iterable, List


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
