"""Acceptance and coverage reporting for synthetic DPO datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.alignment_evidence import (
    build_deterministic_output_validation_summary,
    build_quality_decision_summary,
    deterministic_validation_blockers,
    filter_validation_summary,
    quality_decision_blockers,
)
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
    """Build aggregate and per-dimension DPO acceptance and coverage reporting."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}
    for path in dataset_paths:
        file_rows = read_jsonl(path)
        file_counts[str(path)] = len(file_rows)
        rows.extend(file_rows)

    content = build_dpo_content_summary(rows)
    unique_rows, visible = partition_unique_dpo_rows(rows)
    manifest = _read_run_manifest(run_manifest)
    metadata = manifest.get("metadata", {}) if manifest else {}
    attempted = _non_negative_int(metadata.get("attempted_pairs"), len(rows))
    candidate_pairs = _non_negative_int(metadata.get("candidate_pairs"), attempted)
    accepted = len(unique_rows)
    duplicates = max(_non_negative_int(metadata.get("duplicate_pairs"), 0), visible["duplicate_pairs"])
    rejected = _non_negative_int(metadata.get("rejected_pairs"), 0)
    estimated_tokens = sum(estimate_dpo_tokens(row) for row in unique_rows)
    holdouts = _build_holdout_summary(rows, holdout_registry)
    deterministic_validation = build_deterministic_output_validation_summary(
        row_ids={row["id"] for row in rows},
        manifest=manifest,
        run_manifest_path=Path(run_manifest) if run_manifest is not None else None,
    )
    semantic_adjudication = build_quality_decision_summary(
        row_ids={row["id"] for row in rows},
        manifest=manifest,
        run_manifest_path=Path(run_manifest) if run_manifest is not None else None,
    )
    blockers = _publish_blockers(
        content=content,
        holdouts=holdouts,
        deterministic_validation=deterministic_validation,
        semantic_adjudication=semantic_adjudication,
        manifest_metadata=metadata,
        require_holdout_check=require_holdout_check,
        require_deterministic_validation=run_manifest is not None,
        require_semantic_adjudication=run_manifest is not None,
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
        "preference_dimension_counts": _count_metadata(rows, "preference_dimension"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "failure_modes": _count_metadata(rows, "failure_mode"),
        "content_quality": content,
        "deterministic_output_validation": deterministic_validation,
        "semantic_adjudication": semantic_adjudication,
        "holdouts": holdouts,
        "acceptance": {
            "attempted_pairs": attempted,
            "candidate_pairs": candidate_pairs,
            "accepted_pairs": accepted,
            "estimated_tokens": estimated_tokens,
            "rejected_pairs": rejected,
            "rejection_reason_counts": _count_mapping(metadata.get("rejection_reason_counts")),
            "rejection_diagnostics": _rejection_diagnostics(metadata),
            "duplicate_pairs": duplicates,
            "duplicate_reason_counts": _count_mapping(metadata.get("duplicate_reason_counts")),
            "publish_ready": not blockers,
            "publish_blockers": blockers,
        },
        "preference_dimensions": _build_dimension_reports(
            rows,
            manifest_metadata=metadata,
            holdout_registry=holdout_registry,
            require_holdout_check=require_holdout_check,
            deterministic_validation=deterministic_validation,
            require_deterministic_validation=run_manifest is not None,
            semantic_adjudication=semantic_adjudication,
            require_semantic_adjudication=run_manifest is not None,
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


def _build_dimension_reports(
    rows: list[dict[str, Any]],
    *,
    manifest_metadata: dict[str, Any],
    holdout_registry: HoldoutRegistry | None,
    require_holdout_check: bool,
    deterministic_validation: dict[str, Any],
    require_deterministic_validation: bool,
    semantic_adjudication: dict[str, Any],
    require_semantic_adjudication: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    configured = manifest_metadata.get("candidate_pairs_per_dimension", {})
    dimension_names = {row["metadata"]["preference_dimension"] for row in rows}
    if isinstance(configured, dict):
        dimension_names.update(configured)
    for dimension in sorted(dimension_names):
        dimension_rows = [
            row
            for row in rows
            if row["metadata"]["preference_dimension"] == dimension
        ]
        content = build_dpo_content_summary(dimension_rows)
        unique_rows, visible = partition_unique_dpo_rows(dimension_rows)
        attempted = _dimension_count(
            manifest_metadata,
            "attempted_pairs_per_dimension",
            dimension,
            len(dimension_rows),
        )
        candidates = _dimension_count(
            manifest_metadata, "candidate_pairs_per_dimension", dimension, attempted
        )
        duplicates = max(
            _dimension_count(
                manifest_metadata, "duplicate_pairs_per_dimension", dimension, 0
            ),
            visible["duplicate_pairs"],
        )
        rejected = _dimension_count(
            manifest_metadata, "rejected_pairs_per_dimension", dimension, 0
        )
        holdouts = _build_holdout_summary(dimension_rows, holdout_registry)
        blockers = _publish_blockers(
            content=content,
            holdouts=holdouts,
            deterministic_validation=filter_validation_summary(
                deterministic_validation, {row["id"] for row in dimension_rows}
            ),
            semantic_adjudication=filter_validation_summary(
                semantic_adjudication, {row["id"] for row in dimension_rows}
            ),
            manifest_metadata={},
            require_holdout_check=require_holdout_check,
            require_deterministic_validation=require_deterministic_validation,
            require_semantic_adjudication=require_semantic_adjudication,
        )
        if not unique_rows:
            blockers.append("empty_preference_dimension")
        reports[dimension] = {
            "row_count": len(dimension_rows),
            "task_families": _count_metadata(dimension_rows, "task_family"),
            "interaction_modes": _count_list_metadata(dimension_rows, "interaction_modes"),
            "output_modes": _count_metadata(dimension_rows, "output_mode"),
            "context_modes": _count_metadata(dimension_rows, "context_mode"),
            "template_families": _count_metadata(dimension_rows, "template_family"),
            "difficulty_counts": _count_metadata(
                dimension_rows, "difficulty", stringify_keys=True
            ),
            "failure_modes": _count_metadata(dimension_rows, "failure_mode"),
            "content_quality": content,
            "semantic_adjudication": filter_validation_summary(
                semantic_adjudication, {row["id"] for row in dimension_rows}
            ),
            "holdouts": holdouts,
            "acceptance": {
                "attempted_pairs": attempted,
                "candidate_pairs": candidates,
                "accepted_pairs": len(unique_rows),
                "estimated_tokens": sum(estimate_dpo_tokens(row) for row in unique_rows),
                "rejected_pairs": rejected,
                "rejection_diagnostics": _rejection_diagnostics(
                    manifest_metadata, preference_dimension=dimension
                ),
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
    deterministic_validation: dict[str, Any], manifest_metadata: dict[str, Any],
    semantic_adjudication: dict[str, Any],
    require_holdout_check: bool, require_deterministic_validation: bool,
    require_semantic_adjudication: bool,
) -> list[str]:
    blockers: list[str] = []
    for key, blocker in (("ids", "duplicate_ids"), ("prompts", "duplicate_prompts"), ("triples", "duplicate_triples")):
        if content[key]["duplicate_count"]:
            blockers.append(blocker)
    if holdouts["status"] == "checked" and holdouts["collision_count"]:
        blockers.append("eval_holdout_collisions")
    if require_holdout_check and holdouts["status"] != "checked":
        blockers.append("holdouts_not_checked")
    blockers.extend(
        deterministic_validation_blockers(
            deterministic_validation, required=require_deterministic_validation
        )
    )
    blockers.extend(
        quality_decision_blockers(
            semantic_adjudication, required=require_semantic_adjudication
        )
    )
    if manifest_metadata and manifest_metadata.get("publish_ready") is False:
        blockers.append("run_manifest_not_publish_ready")
    return blockers


def _rejection_diagnostics(
    metadata: dict[str, Any], *, preference_dimension: str | None = None
) -> list[dict[str, Any]]:
    value = metadata.get("rejection_diagnostics", [])
    if not isinstance(value, list):
        return []
    diagnostics = [dict(item) for item in value if isinstance(item, dict)]
    if preference_dimension is not None:
        diagnostics = [
            item
            for item in diagnostics
            if item.get("preference_dimension") == preference_dimension
        ]
    return diagnostics


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


def _read_run_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("metadata", {}), dict):
        raise ValueError(f"DPO run manifest must contain metadata: {manifest_path}")
    return value


def _non_negative_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _dimension_count(
    metadata: dict[str, Any], field: str, dimension: str, fallback: int
) -> int:
    values = metadata.get(field)
    if isinstance(values, dict):
        value = values.get(dimension)
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
