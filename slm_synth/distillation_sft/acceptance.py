"""Dataset-level acceptance helpers for Distillation-SFT."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text
from slm_synth.distillation_sft.response_diversity import (
    build_response_diversity_summary,
    normalize_response_text,
)
from slm_synth.distillation_sft.response_quality import is_response_machine_verified
from slm_synth.distillation_sft.schema import validate_public_row

DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPTS = 1500
DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPT_RATIO = 0.75


def prompt_uniqueness_thresholds_from_env() -> tuple[int, float]:
    min_unique_prompts = int(
        os.getenv(
            "DISTILLATION_SFT_MIN_UNIQUE_PROMPTS",
            str(DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPTS),
        )
    )
    min_unique_ratio = float(
        os.getenv(
            "DISTILLATION_SFT_MIN_UNIQUE_RATIO",
            str(DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPT_RATIO),
        )
    )
    if min_unique_prompts < 0:
        raise ValueError("DISTILLATION_SFT_MIN_UNIQUE_PROMPTS must be non-negative")
    if not 0 <= min_unique_ratio <= 1:
        raise ValueError("DISTILLATION_SFT_MIN_UNIQUE_RATIO must be between 0 and 1")
    return min_unique_prompts, min_unique_ratio


def build_prompt_uniqueness_summary(files: list[Path]) -> dict[str, Any]:
    prompt_counts: dict[str, int] = {}
    row_count = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {file_path} at line {line_number}: {exc}"
                    ) from exc
                validated = validate_public_row(row)
                key = normalize_prompt_text(validated["prompt"])
                prompt_counts[key] = prompt_counts.get(key, 0) + 1
                row_count += 1
    unique_prompt_count = len(prompt_counts)
    return {
        "row_count": row_count,
        "unique_prompt_count": unique_prompt_count,
        "duplicate_prompt_count": row_count - unique_prompt_count,
        "unique_prompt_ratio": unique_prompt_count / row_count if row_count else 0.0,
        "duplicate_examples": [
            prompt for prompt, count in sorted(prompt_counts.items()) if count > 1
        ][:10],
    }


def prompt_uniqueness_blockers(summary: dict[str, Any]) -> list[str]:
    min_unique_prompts, min_unique_ratio = prompt_uniqueness_thresholds_from_env()
    blockers: list[str] = []
    if (
        summary["row_count"] >= min_unique_prompts
        and summary["unique_prompt_count"] < min_unique_prompts
    ):
        blockers.append("insufficient_unique_prompts")
    if summary["row_count"] and summary["unique_prompt_ratio"] < min_unique_ratio:
        blockers.append("low_unique_prompt_ratio")
    return blockers



def require_publish_prompt_uniqueness(
    files: list[Path],
    *,
    min_unique_prompts: int | None = None,
    min_unique_ratio: float | None = None,
) -> dict[str, Any]:
    if min_unique_prompts is None or min_unique_ratio is None:
        env_min_unique_prompts, env_min_unique_ratio = prompt_uniqueness_thresholds_from_env()
        if min_unique_prompts is None:
            min_unique_prompts = env_min_unique_prompts
        if min_unique_ratio is None:
            min_unique_ratio = env_min_unique_ratio
    summary = build_prompt_uniqueness_summary(files)
    failures: list[str] = []
    if summary["row_count"] >= min_unique_prompts and summary["unique_prompt_count"] < min_unique_prompts:
        failures.append(
            f"unique prompts {summary['unique_prompt_count']} below required minimum {min_unique_prompts}"
        )
    if summary["row_count"] and summary["unique_prompt_ratio"] < min_unique_ratio:
        failures.append(
            f"unique prompt ratio {summary['unique_prompt_ratio']:.3f} below required minimum {min_unique_ratio:.3f}"
        )
    if failures:
        examples = ", ".join(repr(example) for example in summary["duplicate_examples"][:3])
        detail = "; ".join(failures)
        if examples:
            detail = f"{detail}; duplicate prompt examples: {examples}"
        raise ValueError(f"distillation-SFT prompt uniqueness gate failed: {detail}")
    return summary


def build_response_cluster_review_summary(files: list[Path]) -> dict[str, Any]:
    diversity = build_response_diversity_summary(files)
    verification_by_response: dict[str, list[bool]] = defaultdict(list)
    for file_path in files:
        signal = file_path.stem.split(".batch", 1)[0]
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {file_path} at line {line_number}: {exc}"
                    ) from exc
                row = validate_public_row(value)
                normalized_response = normalize_response_text(row["response"])
                verification_by_response[normalized_response].append(
                    is_response_machine_verified(signal=signal, row=row)
                )
    classified_clusters: list[dict[str, Any]] = []
    unresolved_clusters: list[dict[str, Any]] = []
    for cluster in diversity["repeated_response_clusters"]:
        checks = verification_by_response[cluster["normalized_response"]]
        automatically_cleared = len(checks) == cluster["count"] and all(checks)
        classified = {
            **cluster,
            "audit_status": "cleared" if automatically_cleared else "review_required",
            "adjudication": "machine_verified" if automatically_cleared else None,
        }
        classified_clusters.append(classified)
        if not automatically_cleared:
            unresolved_clusters.append(classified)
    return {
        "repeated_cluster_count": len(classified_clusters),
        "automatically_cleared_cluster_count": len(classified_clusters) - len(unresolved_clusters),
        "unresolved_cluster_count": len(unresolved_clusters),
        "clusters": classified_clusters,
        "unresolved_clusters": unresolved_clusters,
    }


def require_publish_resolved_response_clusters(files: list[Path]) -> dict[str, Any]:
    summary = build_response_cluster_review_summary(files)
    unresolved = summary["unresolved_clusters"]
    if unresolved:
        examples = []
        for cluster in unresolved[:3]:
            row_ids = [member["id"] for member in cluster["members"][:5]]
            examples.append(
                f"{cluster['response_fingerprint'][:12]} count={cluster['count']} rows={','.join(row_ids)}"
            )
        raise ValueError(
            "distillation-SFT response cluster gate failed: "
            f"{len(unresolved)} unresolved repeated-response cluster(s) require review; "
            + "; ".join(examples)
        )
    return summary
