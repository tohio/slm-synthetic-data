"""Coverage reporting helpers for synthetic SFT datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.sft.acceptance import build_sft_content_summary, partition_unique_sft_rows
from slm_synth.sft.io import read_jsonl
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def build_coverage_report(
    paths: list[str | Path],
    *,
    holdout_registry: HoldoutRegistry | None = None,
    run_manifest: str | Path | None = None,
    require_holdout_check: bool = True,
) -> dict[str, Any]:
    """Build aggregate and per-family SFT acceptance and coverage reporting."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}

    for path in dataset_paths:
        file_rows = read_jsonl(path)
        file_counts[str(path)] = len(file_rows)
        rows.extend(file_rows)

    content = build_sft_content_summary(rows)
    unique_rows, visible_acceptance = partition_unique_sft_rows(rows)
    manifest_metadata = _read_run_manifest_metadata(run_manifest)
    accepted_target = manifest_metadata.get("accepted_target", {})
    if not isinstance(accepted_target, dict):
        accepted_target = {}

    attempted_rows = _non_negative_int(
        accepted_target.get("attempted"),
        manifest_metadata.get("attempted_rows"),
        len(rows),
    )
    duplicate_rows = max(
        _non_negative_int(manifest_metadata.get("duplicate_rows"), 0),
        visible_acceptance["duplicate_rows"],
    )
    rejected_rows = _non_negative_int(manifest_metadata.get("rejected_rows"), 0)
    accepted_rows = len(unique_rows)
    target_rows = _non_negative_int(
        accepted_target.get("target"),
        manifest_metadata.get("planned_rows"),
        attempted_rows,
    )
    remaining_rows = max(
        _non_negative_int(
            accepted_target.get("remaining"),
            manifest_metadata.get("remaining_rows"),
            0,
        ),
        target_rows - accepted_rows,
    )

    holdouts = _build_holdout_summary(rows, holdout_registry)
    blockers = _publish_blockers(
        content=content,
        holdouts=holdouts,
        remaining_rows=remaining_rows,
        manifest_metadata=manifest_metadata,
        require_holdout_check=require_holdout_check,
    )

    return {
        "dataset_type": "sft",
        "row_count": len(rows),
        "files": file_counts,
        "categories": _count_metadata(rows, "category"),
        "eval_families": _count_metadata(rows, "eval_family"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "content_uniqueness": content,
        "holdouts": holdouts,
        "acceptance": {
            "attempted_rows": attempted_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "duplicate_rows": duplicate_rows,
            "remaining_rows": remaining_rows,
            "publish_ready": not blockers,
            "publish_blockers": blockers,
        },
        "families": _build_family_reports(
            rows,
            manifest_metadata=manifest_metadata,
            holdout_registry=holdout_registry,
            require_holdout_check=require_holdout_check,
        ),
    }


def require_publish_ready_report(report: dict[str, Any], *, artifact_name: str = "SFT") -> None:
    """Reject an SFT artifact whose acceptance report contains publish blockers."""
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{artifact_name} coverage report is missing acceptance reporting")
    blockers = acceptance.get("publish_blockers")
    if acceptance.get("publish_ready") is not True or not isinstance(blockers, list) or blockers:
        detail = ", ".join(str(blocker) for blocker in blockers) if isinstance(blockers, list) else "unknown"
        raise ValueError(f"{artifact_name} acceptance report is not publish-ready: {detail}")


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
    """Write a coverage report JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


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


def _count_metadata(
    rows: list[dict[str, Any]],
    field: str,
    *,
    stringify_keys: bool = False,
) -> dict[str, int]:
    counter = Counter(row["metadata"][field] for row in rows)
    if stringify_keys:
        return {str(key): counter[key] for key in sorted(counter)}
    return {key: counter[key] for key in sorted(counter)}


def _build_family_reports(
    rows: list[dict[str, Any]],
    *,
    manifest_metadata: dict[str, Any],
    holdout_registry: HoldoutRegistry | None,
    require_holdout_check: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for family in sorted({row["metadata"]["eval_family"] for row in rows}):
        family_rows = [row for row in rows if row["metadata"]["eval_family"] == family]
        content = build_sft_content_summary(family_rows)
        unique_rows, visible_acceptance = partition_unique_sft_rows(family_rows)
        attempted = _family_count(manifest_metadata, "attempted_rows_per_family", family, len(family_rows))
        duplicates = max(
            _family_count(manifest_metadata, "duplicate_rows_per_family", family, 0),
            visible_acceptance["duplicate_rows"],
        )
        rejected = _family_count(
            manifest_metadata,
            "rejected_rows_per_family",
            family,
            max(attempted - len(unique_rows) - duplicates, 0),
        )
        target = _family_count(manifest_metadata, "rows_per_family", family, attempted)
        remaining = max(target - len(unique_rows), 0)
        holdouts = _build_holdout_summary(family_rows, holdout_registry)
        blockers = _publish_blockers(
            content=content,
            holdouts=holdouts,
            remaining_rows=remaining,
            manifest_metadata={},
            require_holdout_check=require_holdout_check,
        )
        reports[family] = {
            "row_count": len(family_rows),
            "categories": _count_metadata(family_rows, "category"),
            "template_families": _count_metadata(family_rows, "template_family"),
            "difficulty_counts": _count_metadata(family_rows, "difficulty", stringify_keys=True),
            "content_uniqueness": content,
            "holdouts": holdouts,
            "acceptance": {
                "attempted_rows": attempted,
                "accepted_rows": len(unique_rows),
                "rejected_rows": rejected,
                "duplicate_rows": duplicates,
                "remaining_rows": remaining,
                "publish_ready": not blockers,
                "publish_blockers": blockers,
            },
        }
    return reports


def _build_holdout_summary(
    rows: list[dict[str, Any]],
    registry: HoldoutRegistry | None,
) -> dict[str, Any]:
    if registry is None:
        return {"status": "not_checked", "collision_count": None, "collision_ids": []}

    collision_ids: list[str] = []
    for row in rows:
        if any(
            message["role"] == "user" and registry.contains_prompt(message["content"])
            for message in row["messages"]
        ):
            collision_ids.append(row["id"])
    return {
        "status": "checked",
        "collision_count": len(collision_ids),
        "collision_ids": collision_ids,
    }


def _publish_blockers(
    *,
    content: dict[str, Any],
    holdouts: dict[str, Any],
    remaining_rows: int,
    manifest_metadata: dict[str, Any],
    require_holdout_check: bool,
) -> list[str]:
    blockers: list[str] = []
    for key, blocker in (
        ("ids", "duplicate_ids"),
        ("prompts", "duplicate_prompts"),
        ("conversations", "duplicate_conversations"),
    ):
        if content[key]["duplicate_count"]:
            blockers.append(blocker)
    if holdouts["status"] == "checked" and holdouts["collision_count"]:
        blockers.append("eval_holdout_collisions")
    if require_holdout_check and holdouts["status"] != "checked":
        blockers.append("holdouts_not_checked")
    if remaining_rows:
        blockers.append("accepted_target_underfilled")
    if manifest_metadata and manifest_metadata.get("publish_ready") is False:
        blockers.append("run_manifest_not_publish_ready")
    return blockers


def _read_run_manifest_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SFT run manifest must contain a JSON object: {manifest_path}")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"SFT run manifest metadata must contain an object: {manifest_path}")
    return metadata


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
