"""Coverage reporting helpers for response-distillation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_synth.distillation.card import load_run_manifest


def build_coverage_report(run_manifest_path: str | Path) -> dict[str, Any]:
    """Build a compact coverage report from a distillation run manifest."""
    manifest = load_run_manifest(run_manifest_path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("run manifest datasets must be a list")

    signal_counts: dict[str, int] = {}
    dataset_paths: dict[str, str] = {}
    manifest_paths: dict[str, str] = {}
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
            dataset_paths[signal] = str(Path(str(dataset_path)))
        manifest_path = dataset.get("manifest_path")
        if manifest_path is not None:
            manifest_paths[signal] = str(Path(str(manifest_path)))

    return {
        "dataset_type": "distillation",
        "generation_run": _require_non_empty_string(manifest.get("generation_run"), "generation_run"),
        "teacher_model": _require_non_empty_string(manifest.get("teacher_model"), "teacher_model"),
        "teacher_provider": _require_non_empty_string(manifest.get("teacher_provider"), "teacher_provider"),
        "token_target": manifest.get("token_target"),
        "row_count": total_rows,
        "signals": {signal: signal_counts[signal] for signal in sorted(signal_counts)},
        "dataset_paths": {signal: dataset_paths[signal] for signal in sorted(dataset_paths)},
        "manifest_paths": {signal: manifest_paths[signal] for signal in sorted(manifest_paths)},
    }


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
    """Write a distillation coverage report JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
