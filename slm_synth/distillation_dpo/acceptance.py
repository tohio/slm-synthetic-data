"""Dataset-level acceptance reporting for Distillation-DPO."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.distillation_dpo.pair_quality import (
    filter_pairs_by_quality,
    normalized_preference_triple_fingerprint,
    normalized_prompt_fingerprint,
)


def build_dataset_acceptance_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_categories: Mapping[str, int] | None = None,
    expected_failure_modes: Mapping[str, int] | None = None,
    attempted_pairs: int | None = None,
    attempted_rejection_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Audit accepted rows against uniqueness, quality, and coverage contracts."""
    materialized_rows = list(rows)
    accepted_rows, audit = filter_pairs_by_quality(
        family="teacher_response_preference",
        rows=materialized_rows,
    )
    categories = _metadata_counts(accepted_rows, "category")
    failure_modes = _metadata_counts(accepted_rows, "failure_mode")
    expected_category_counts = _sorted_counts(expected_categories or categories)
    expected_failure_mode_counts = _sorted_counts(expected_failure_modes or failure_modes)
    row_count = len(materialized_rows)
    unique_prompt_count = len({normalized_prompt_fingerprint(row) for row in materialized_rows})
    unique_triple_count = len(
        {normalized_preference_triple_fingerprint(row) for row in materialized_rows}
    )
    uniqueness_satisfied = (
        unique_prompt_count == row_count
        and unique_triple_count == row_count
        and audit.accepted_pairs == row_count
    )
    coverage_satisfied = (
        categories == expected_category_counts
        and failure_modes == expected_failure_mode_counts
    )
    rejection_reasons = _sorted_counts(attempted_rejection_reasons or audit.rejection_reasons)
    attempted = attempted_pairs if attempted_pairs is not None else audit.checked_pairs
    rejected = max(attempted - audit.accepted_pairs, 0)
    return {
        "attempted_pairs": attempted,
        "rejected_pairs": rejected,
        "duplicate_prompt_pairs": rejection_reasons.get("duplicate_prompt", 0),
        "duplicate_triple_pairs": rejection_reasons.get("duplicate_preference_triple", 0),
        "accepted_pairs": audit.accepted_pairs,
        "remaining_pairs": max(sum(expected_category_counts.values()) - audit.accepted_pairs, 0),
        "unique_prompt_count": unique_prompt_count,
        "unique_triple_count": unique_triple_count,
        "categories": categories,
        "failure_modes": failure_modes,
        "expected_categories": expected_category_counts,
        "expected_failure_modes": expected_failure_mode_counts,
        "rejection_reasons": rejection_reasons,
        "uniqueness_satisfied": uniqueness_satisfied,
        "coverage_satisfied": coverage_satisfied,
        "publish_ready": uniqueness_satisfied and coverage_satisfied,
    }


def require_dataset_acceptance(report: Mapping[str, Any], *, artifact_name: str) -> None:
    """Reject an artifact whose dataset-level DPO contract is not satisfied."""
    if report.get("publish_ready") is True:
        return
    raise ValueError(
        f"{artifact_name} fails the Distillation-DPO dataset acceptance contract: "
        f"unique_prompts={report.get('unique_prompt_count')} "
        f"unique_triples={report.get('unique_triple_count')} "
        f"accepted={report.get('accepted_pairs')} "
        f"coverage_satisfied={report.get('coverage_satisfied')}"
    )


def _metadata_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        if isinstance(metadata, Mapping) and isinstance(metadata.get(field), str):
            counts[str(metadata[field])] += 1
    return dict(sorted(counts.items()))


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items())}
