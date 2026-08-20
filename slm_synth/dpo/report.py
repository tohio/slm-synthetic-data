"""Acceptance and coverage reporting for synthetic DPO datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.alignment_tokens import estimate_dpo_tokens
from slm_synth.dpo.acceptance import build_dpo_content_summary, partition_unique_dpo_rows
from slm_synth.dpo.io import read_jsonl
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def build_coverage_report(
    paths: list[str | Path],
    *,
    holdout_registry: HoldoutRegistry | None = None,
    run_manifest: str | Path | None = None,
    require_holdout_check: bool = True,
) -> dict[str, Any]:
    """Build aggregate and per-family DPO acceptance and coverage reporting."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}
    for path in dataset_paths:
        file_rows = read_jsonl(path)
        file_counts[str(path)] = len(file_rows)
        rows.extend(file_rows)

    content = build_dpo_content_summary(rows)
    unique_rows, visible = partition_unique_dpo_rows(rows)
    metadata = _read_run_manifest_metadata(run_manifest)
    attempted = _non_negative_int(metadata.get("attempted_pairs"), len(rows))
    candidate_pairs = _non_negative_int(metadata.get("candidate_pairs"), attempted)
    accepted = len(unique_rows)
    duplicates = max(_non_negative_int(metadata.get("duplicate_pairs"), 0), visible["duplicate_pairs"])
    rejected = _non_negative_int(metadata.get("rejected_pairs"), 0)
    estimated_tokens = sum(estimate_dpo_tokens(row) for row in unique_rows)
    holdouts = _build_holdout_summary(rows, holdout_registry)
    blockers = _publish_blockers(
        content=content,
        holdouts=holdouts,
        manifest_metadata=metadata,
        require_holdout_check=require_holdout_check,
    )
    if not unique_rows:
        blockers.append("empty_dataset")
    return {
        "dataset_type": "dpo",
        "row_count": len(rows),
        "files": file_counts,
        "task_families": _count_metadata(rows, "task_family"),
        "interaction_modes": _count_list_metadata(rows, "interaction_modes"),
        "output_modes": _count_metadata(rows, "output_mode"),
        "context_modes": _count_metadata(rows, "context_mode"),
        "preference_dimensions": _count_metadata(rows, "preference_dimension"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "failure_modes": _count_metadata(rows, "failure_mode"),
        "content_quality": content,
        "holdouts": holdouts,
        "acceptance": {
            "attempted_pairs": attempted,
            "candidate_pairs": candidate_pairs,
            "accepted_pairs": accepted,
            "estimated_tokens": estimated_tokens,
            "rejected_pairs": rejected,
            "rejection_reason_counts": _count_mapping(metadata.get("rejection_reason_counts")),
            "duplicate_pairs": duplicates,
            "duplicate_reason_counts": _count_mapping(metadata.get("duplicate_reason_counts")),
            "publish_ready": not blockers,
            "publish_blockers": blockers,
        },
        "families": _build_family_reports(
            rows,
            manifest_metadata=metadata,
            holdout_registry=holdout_registry,
            require_holdout_check=require_holdout_check,
        ),
    }


def require_publish_ready_report(report: dict[str, Any], *, artifact_name: str = "DPO") -> None:
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{artifact_name} coverage report is missing acceptance reporting")
    blockers = acceptance.get("publish_blockers")
    if acceptance.get("publish_ready") is not True or not isinstance(blockers, list) or blockers:
        detail = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else "unknown"
        raise ValueError(f"{artifact_name} acceptance report is not publish-ready: {detail}")


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _build_family_reports(
    rows: list[dict[str, Any]],
    *,
    manifest_metadata: dict[str, Any],
    holdout_registry: HoldoutRegistry | None,
    require_holdout_check: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    configured = manifest_metadata.get("candidate_pairs_per_dimension", {})
    family_names = {row["metadata"]["preference_dimension"] for row in rows}
    if isinstance(configured, dict):
        family_names.update(configured)
    for family in sorted(family_names):
        family_rows = [row for row in rows if row["metadata"]["preference_dimension"] == family]
        content = build_dpo_content_summary(family_rows)
        unique_rows, visible = partition_unique_dpo_rows(family_rows)
        attempted = _family_count(manifest_metadata, "attempted_pairs_per_dimension", family, len(family_rows))
        candidates = _family_count(
            manifest_metadata, "candidate_pairs_per_dimension", family, attempted
        )
        duplicates = max(
            _family_count(manifest_metadata, "duplicate_pairs_per_dimension", family, 0),
            visible["duplicate_pairs"],
        )
        rejected = _family_count(manifest_metadata, "rejected_pairs_per_dimension", family, 0)
        holdouts = _build_holdout_summary(family_rows, holdout_registry)
        blockers = _publish_blockers(
            content=content,
            holdouts=holdouts,
            manifest_metadata={},
            require_holdout_check=require_holdout_check,
        )
        reports[family] = {
            "row_count": len(family_rows),
            "task_families": _count_metadata(family_rows, "task_family"),
            "interaction_modes": _count_list_metadata(family_rows, "interaction_modes"),
            "output_modes": _count_metadata(family_rows, "output_mode"),
            "context_modes": _count_metadata(family_rows, "context_mode"),
            "template_families": _count_metadata(family_rows, "template_family"),
            "difficulty_counts": _count_metadata(family_rows, "difficulty", stringify_keys=True),
            "failure_modes": _count_metadata(family_rows, "failure_mode"),
            "content_quality": content,
            "holdouts": holdouts,
            "acceptance": {
                "attempted_pairs": attempted,
                "candidate_pairs": candidates,
                "accepted_pairs": len(unique_rows),
                "estimated_tokens": sum(estimate_dpo_tokens(row) for row in unique_rows),
                "rejected_pairs": rejected,
                "duplicate_pairs": duplicates,
                "publish_ready": not blockers,
                "publish_blockers": blockers,
            },
        }
    return reports


def _build_holdout_summary(rows: list[dict[str, Any]], registry: HoldoutRegistry | None) -> dict[str, Any]:
    if registry is None:
        return {"status": "not_checked", "collision_count": None, "collision_ids": []}
    collision_ids = [
        row["id"]
        for row in rows
        if any(message["role"] == "user" and registry.contains_prompt(message["content"]) for message in row["prompt"])
    ]
    return {"status": "checked", "collision_count": len(collision_ids), "collision_ids": collision_ids}


def _publish_blockers(
    *, content: dict[str, Any], holdouts: dict[str, Any],
    manifest_metadata: dict[str, Any], require_holdout_check: bool,
) -> list[str]:
    blockers: list[str] = []
    for key, blocker in (("ids", "duplicate_ids"), ("prompts", "duplicate_prompts"), ("triples", "duplicate_triples")):
        if content[key]["duplicate_count"]:
            blockers.append(blocker)
    if holdouts["status"] == "checked" and holdouts["collision_count"]:
        blockers.append("eval_holdout_collisions")
    if require_holdout_check and holdouts["status"] != "checked":
        blockers.append("holdouts_not_checked")
    if manifest_metadata and manifest_metadata.get("publish_ready") is False:
        blockers.append("run_manifest_not_publish_ready")
    return blockers


def _resolve_jsonl_paths(paths: list[str | Path]) -> list[Path]:
    if not paths:
        raise ValueError("at least one input path is required")
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file()))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"input path does not exist: {path}")
    if not resolved:
        raise ValueError("no JSONL dataset files found")
    return resolved


def _count_metadata(rows: list[dict[str, Any]], field: str, *, stringify_keys: bool = False) -> dict[str, int]:
    counter = Counter(row["metadata"][field] for row in rows)
    if stringify_keys:
        return {str(key): counter[key] for key in sorted(counter)}
    return {key: counter[key] for key in sorted(counter)}


def _count_list_metadata(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(item for row in rows for item in row["metadata"][field])
    return {key: counter[key] for key in sorted(counter)}


def _read_run_manifest_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("metadata", {}), dict):
        raise ValueError(f"DPO run manifest must contain metadata: {manifest_path}")
    return value.get("metadata", {})


def _non_negative_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _family_count(metadata: dict[str, Any], field: str, family: str, fallback: int) -> int:
    values = metadata.get(field)
    if isinstance(values, dict):
        value = values.get(family)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return fallback


def _count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: count for key, count in sorted(value.items())
        if isinstance(key, str) and isinstance(count, int) and not isinstance(count, bool) and count >= 0
    }
