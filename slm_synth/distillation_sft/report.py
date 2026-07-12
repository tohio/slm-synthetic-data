"""Coverage reporting helpers for response-distillation runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.card import load_run_manifest
from slm_synth.distillation_sft.schema import validate_public_row


def build_coverage_report(run_manifest_path: str | Path) -> dict[str, Any]:
    """Build a compact coverage report from a distillation run manifest."""
    manifest = load_run_manifest(run_manifest_path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("run manifest datasets must be a list")

    signal_counts: dict[str, int] = {}
    dataset_paths: dict[str, str] = {}
    manifest_paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    total_rows = 0
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each run manifest dataset must be an object")
        signal = _require_non_empty_string(dataset.get("signal"), "signal")
        row_count = dataset.get("row_count")
        if not isinstance(row_count, int) or row_count < 0:
            raise ValueError("dataset row_count must be a non-negative integer")
        if signal in signal_counts:
            raise ValueError(f"duplicate signal in run manifest datasets: {signal}")
        signal_counts[signal] = row_count
        total_rows += row_count

        dataset_path = dataset.get("dataset_path")
        if dataset_path is not None:
            resolved_dataset_path = Path(str(dataset_path))
            dataset_paths[signal] = str(resolved_dataset_path)
            if resolved_dataset_path.is_file():
                rows.extend(_read_public_rows(resolved_dataset_path))
        manifest_path = dataset.get("manifest_path")
        if manifest_path is not None:
            manifest_paths[signal] = str(Path(str(manifest_path)))

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "dataset_type": "distillation",
        "generation_run": _require_non_empty_string(manifest.get("generation_run"), "generation_run"),
        "teacher_model": _require_non_empty_string(manifest.get("teacher_model"), "teacher_model"),
        "teacher_provider": _require_non_empty_string(manifest.get("teacher_provider"), "teacher_provider"),
        "token_target": manifest.get("token_target"),
        "target_rows": metadata.get("target_rows"),
        "planned_prompt_rows": metadata.get("planned_prompt_rows"),
        "accepted_rows": metadata.get("accepted_rows", total_rows),
        "rejected_rows": metadata.get("rejected_rows"),
        "row_count": total_rows,
        "signals": {signal: signal_counts[signal] for signal in sorted(signal_counts)},
        "rows_per_signal": _sorted_mapping(metadata.get("rows_per_signal")),
        "categories": _count_metadata(rows, "category"),
        "eval_families": _count_metadata(rows, "eval_family"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty"),
        "dataset_paths": {signal: dataset_paths[signal] for signal in sorted(dataset_paths)},
        "manifest_paths": {signal: manifest_paths[signal] for signal in sorted(manifest_paths)},
    }


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
    """Write a distillation coverage report JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _sorted_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): value[key] for key in sorted(value)}


def _read_public_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
            rows.append(validate_public_row(value))
    return rows


def _count_metadata(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter("null" if row["metadata"][field] is None else str(row["metadata"][field]) for row in rows)
    return {key: counter[key] for key in sorted(counter)}


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
