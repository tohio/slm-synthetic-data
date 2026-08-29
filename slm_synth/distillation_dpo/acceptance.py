"""Dataset-level acceptance reporting for Distillation-DPO."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any

from slm_synth.distillation_dpo.pair_quality import (
    filter_pairs_by_quality,
    normalize_pair_text,
    normalized_preference_triple_fingerprint,
    normalized_prompt_fingerprint,
)
from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row


def build_dataset_acceptance_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_categories: Mapping[str, int] | None = None,
    expected_failure_modes: Mapping[str, int] | None = None,
    attempted_pairs: int | None = None,
    attempted_rejection_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Audit final accepted rows against uniqueness and coverage contracts.

    The generation pipeline is the acceptance authority. The legacy deterministic
    pair-quality pass is retained as diagnostics only, matching the generic DPO
    report/push lifecycle instead of applying a second publish-time acceptance gate.
    """
    materialized_rows = [validate_distillation_dpo_row(row) for row in rows]
    _, pair_quality_audit = filter_pairs_by_quality(
        family="teacher_response_preference",
        rows=materialized_rows,
    )
    categories = _metadata_counts(materialized_rows, "category")
    failure_modes = _metadata_counts(materialized_rows, "failure_mode")
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
    )
    coverage_satisfied = (
        categories == expected_category_counts
        and failure_modes == expected_failure_mode_counts
    )
    rejection_reasons = _sorted_counts(attempted_rejection_reasons or {})
    attempted = attempted_pairs if attempted_pairs is not None else row_count
    rejected = max(attempted - row_count, 0)
    return {
        "attempted_pairs": attempted,
        "rejected_pairs": rejected,
        "duplicate_prompt_pairs": row_count - unique_prompt_count,
        "duplicate_triple_pairs": row_count - unique_triple_count,
        "accepted_pairs": row_count,
        "remaining_pairs": max(sum(expected_category_counts.values()) - row_count, 0),
        "unique_prompt_count": unique_prompt_count,
        "unique_triple_count": unique_triple_count,
        "categories": categories,
        "failure_modes": failure_modes,
        "expected_categories": expected_category_counts,
        "expected_failure_modes": expected_failure_mode_counts,
        "rejection_reasons": rejection_reasons,
        "pair_quality_diagnostics": pair_quality_audit.to_dict(),
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


def build_response_pattern_report(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report response repetition and pair-shape diagnostics without gating rows."""
    validated_rows = [validate_distillation_dpo_row(row) for row in rows]
    similarities = [
        {
            "id": row["id"],
            "score": _chosen_rejected_similarity(row),
        }
        for row in validated_rows
    ]
    pattern_rows: dict[str, list[str]] = {}
    for row in validated_rows:
        pattern = _negative_pattern(row)
        pattern_rows.setdefault(pattern, []).append(row["id"])
    return {
        "chosen_responses": _response_repetition(validated_rows, "chosen"),
        "rejected_responses": _response_repetition(validated_rows, "rejected"),
        "chosen_rejected_similarity": {
            "minimum": min((item["score"] for item in similarities), default=0.0),
            "mean": (
                sum(item["score"] for item in similarities) / len(similarities)
                if similarities
                else 0.0
            ),
            "maximum": max((item["score"] for item in similarities), default=0.0),
            "at_or_above_0_90": sum(item["score"] >= 0.90 for item in similarities),
            "at_or_above_0_98": sum(item["score"] >= 0.98 for item in similarities),
            "rows_at_or_above_0_90": [
                item for item in similarities if item["score"] >= 0.90
            ],
        },
        "negative_patterns": {
            "counts": {
                pattern: len(row_ids)
                for pattern, row_ids in sorted(pattern_rows.items())
            },
            "row_ids": {
                pattern: row_ids
                for pattern, row_ids in sorted(pattern_rows.items())
            },
            "repeated_pattern_count": sum(
                len(row_ids) > 1 for row_ids in pattern_rows.values()
            ),
            "maximum_repetition": max(
                (len(row_ids) for row_ids in pattern_rows.values()),
                default=0,
            ),
        },
        "policy": {
            "repeated_response_clusters_are_diagnostic": True,
            "similarity_is_diagnostic": True,
            "negative_patterns_are_diagnostic": True,
            "automatic_semantic_judge": False,
        },
    }


def _response_repetition(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in rows:
        response = _assistant_text(row[field])
        fingerprint = _text_fingerprint(response)
        cluster = clusters.setdefault(
            fingerprint,
            {
                "response_fingerprint": fingerprint,
                "response": response,
                "row_ids": [],
                "prompts": [],
                "categories": set(),
                "failure_modes": set(),
            },
        )
        cluster["row_ids"].append(row["id"])
        cluster["prompts"].append(_last_user_text(row["prompt"]))
        cluster["categories"].add(row["metadata"]["category"])
        cluster["failure_modes"].add(row["metadata"]["failure_mode"])

    repeated_clusters = []
    for fingerprint in sorted(clusters):
        cluster = clusters[fingerprint]
        if len(cluster["row_ids"]) <= 1:
            continue
        repeated_clusters.append(
            {
                **cluster,
                "count": len(cluster["row_ids"]),
                "categories": sorted(cluster["categories"]),
                "failure_modes": sorted(cluster["failure_modes"]),
            }
        )
    total = len(rows)
    unique = len(clusters)
    return {
        "total": total,
        "unique": unique,
        "duplicate_count": total - unique,
        "unique_ratio": unique / total if total else 1.0,
        "repeated_cluster_count": len(repeated_clusters),
        "maximum_repetition": max(
            (len(cluster["row_ids"]) for cluster in clusters.values()),
            default=0,
        ),
        "repeated_clusters": repeated_clusters,
    }


def _chosen_rejected_similarity(row: Mapping[str, Any]) -> float:
    chosen = normalize_pair_text(_assistant_text(row["chosen"]))
    rejected = normalize_pair_text(_assistant_text(row["rejected"]))
    return SequenceMatcher(None, chosen, rejected, autojunk=False).ratio()


def _negative_pattern(row: Mapping[str, Any]) -> str:
    chosen = normalize_pair_text(_assistant_text(row["chosen"]))
    rejected = normalize_pair_text(_assistant_text(row["rejected"]))
    if chosen and chosen in rejected:
        return "chosen_verbatim_with_extra"
    if rejected and rejected in chosen:
        return "truncated_chosen"
    if _is_number(chosen) and _is_number(rejected):
        return "numeric_substitution"
    if rejected.startswith(("i cannot", "i can't", "sorry", "as an ai")):
        return "refusal_or_disclaimer"
    if "```" in _assistant_text(row["rejected"]):
        return "code_wrapper_or_fence"
    return "other"


def _assistant_text(messages: Any) -> str:
    return "\n".join(
        message["content"]
        for message in messages
        if isinstance(message, Mapping) and isinstance(message.get("content"), str)
    )


def _last_user_text(messages: Any) -> str:
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _text_fingerprint(value: str) -> str:
    return hashlib.sha256(normalize_pair_text(value).encode("utf-8")).hexdigest()


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return bool(value)


def _metadata_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        if isinstance(metadata, Mapping) and isinstance(metadata.get(field), str):
            counts[str(metadata[field])] += 1
    return dict(sorted(counts.items()))


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items())}
