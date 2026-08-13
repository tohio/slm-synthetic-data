"""Coverage reporting helpers for distillation-DPO datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.distillation_dpo.acceptance import (
    build_dataset_acceptance_report,
    build_response_pattern_report,
)
from slm_synth.distillation_dpo.io import DATASET_TYPE, read_jsonl
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def build_coverage_report(
    paths: list[str | Path],
    *,
    holdout_registry: HoldoutRegistry | None = None,
    require_holdout_check: bool = True,
) -> dict[str, Any]:
    """Build a compact coverage report for distillation-DPO JSONL files."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}
    for path in dataset_paths:
        file_rows = read_jsonl(path)
        file_counts[str(path)] = len(file_rows)
        rows.extend(file_rows)

    acceptance = build_dataset_acceptance_report(rows)
    holdouts = _build_holdout_summary(rows, holdout_registry)
    blockers: list[str] = []
    if acceptance["publish_ready"] is not True:
        blockers.append("dataset_acceptance_failed")
    if holdouts["status"] == "checked" and holdouts["collision_count"]:
        blockers.append("eval_holdout_collisions")
    if require_holdout_check and holdouts["status"] != "checked":
        blockers.append("holdouts_not_checked")
    acceptance["publish_blockers"] = blockers
    acceptance["publish_ready"] = not blockers
    return {
        "dataset_type": DATASET_TYPE,
        "row_count": len(rows),
        "files": file_counts,
        "categories": _count_metadata(rows, "category"),
        "eval_families": _count_metadata(rows, "eval_family"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "failure_modes": _count_metadata(rows, "failure_mode"),
        "response_patterns": build_response_pattern_report(rows),
        "holdouts": holdouts,
        "dataset_acceptance": acceptance,
    }


def require_publish_ready_report(
    report: dict[str, Any],
    *,
    artifact_name: str = "Distillation DPO",
) -> None:
    acceptance = report.get("dataset_acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{artifact_name} coverage report is missing dataset acceptance")
    blockers = acceptance.get("publish_blockers")
    if acceptance.get("publish_ready") is not True or not isinstance(blockers, list) or blockers:
        detail = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else "unknown"
        raise ValueError(f"{artifact_name} coverage report is not publish-ready: {detail}")


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
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


def _count_metadata(rows: list[dict[str, Any]], field: str, *, stringify_keys: bool = False) -> dict[str, int]:
    counter = Counter(row["metadata"][field] for row in rows)
    if stringify_keys:
        return {str(key): counter[key] for key in sorted(counter)}
    return {key: counter[key] for key in sorted(counter)}


def _build_holdout_summary(
    rows: list[dict[str, Any]],
    registry: HoldoutRegistry | None,
) -> dict[str, Any]:
    if registry is None:
        return {"status": "not_checked", "collision_count": None, "collision_ids": []}
    collision_ids = [
        row["id"]
        for row in rows
        if any(
            message["role"] == "user" and registry.contains_prompt(message["content"])
            for message in row["prompt"]
        )
    ]
    return {
        "status": "checked",
        "collision_count": len(collision_ids),
        "collision_ids": collision_ids,
    }
