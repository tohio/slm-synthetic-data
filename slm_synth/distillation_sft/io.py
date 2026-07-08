"""JSONL and manifest writing helpers for response-distillation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.schema import validate_public_row
from slm_synth.distillation_sft.signals import validate_signal


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Write validated public distillation rows to JSONL and return row count."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validated = validate_public_row(row)
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_signal_dataset(
    *,
    signal: str,
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Write one signal-specific public distillation dataset.

    Public rows contain only id, prompt, reasoning, and response. Signal names
    are represented by the dataset file path, not repeated inside each row.
    """
    normalized_signal = validate_signal(signal)
    output_root = Path(output_dir)
    output_path = output_root / (filename or f"{normalized_signal}.jsonl")
    write_jsonl(rows, output_path)
    return output_path


def write_manifest(
    *,
    manifest_path: str | Path,
    signal: str,
    dataset_path: str | Path,
    row_count: int,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    token_target: str | int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local-only manifest for run/provenance details.

    The manifest is intentionally separate from public training rows. It may
    include provider, teacher model, run id, token target, cost/retry summaries,
    and other local metadata.
    """
    normalized_signal = validate_signal(signal)
    if not isinstance(teacher_model, str) or not teacher_model.strip():
        raise ValueError("teacher_model must be a non-empty string")
    if teacher_provider.strip().lower() != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    if row_count < 0:
        raise ValueError("row_count must be non-negative")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signal": normalized_signal,
        "dataset_path": str(Path(dataset_path)),
        "row_count": row_count,
        "teacher_model": teacher_model,
        "teacher_provider": "openrouter",
        "generation_run": generation_run,
        "token_target": token_target,
        "metadata": dict(metadata or {}),
    }

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def write_run_manifest(
    *,
    manifest_path: str | Path,
    generation_run: str,
    teacher_model: str,
    teacher_provider: str,
    token_target: str | int | None,
    datasets: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local-only manifest summarizing a multi-signal run.

    This manifest ties together the signal-specific datasets and manifests from
    one generation run. It is intentionally local provenance data, not part of
    the public training row schema.
    """
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    if not isinstance(teacher_model, str) or not teacher_model.strip():
        raise ValueError("teacher_model must be a non-empty string")
    if teacher_provider.strip().lower() != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")

    normalized_datasets: list[dict[str, Any]] = []
    seen_signals: set[str] = set()
    total_rows = 0
    for dataset in datasets:
        signal = validate_signal(dataset.get("signal"))
        if signal in seen_signals:
            raise ValueError(f"duplicate signal in run manifest datasets: {signal}")
        seen_signals.add(signal)

        row_count = dataset.get("row_count")
        if not isinstance(row_count, int) or row_count < 0:
            raise ValueError("dataset row_count must be a non-negative integer")

        dataset_path = dataset.get("dataset_path")
        manifest_item_path = dataset.get("manifest_path")
        if dataset_path is None:
            raise ValueError("dataset_path is required for each run manifest dataset")
        if manifest_item_path is None:
            raise ValueError("manifest_path is required for each run manifest dataset")

        normalized_datasets.append(
            {
                "signal": signal,
                "dataset_path": str(Path(dataset_path)),
                "manifest_path": str(Path(manifest_item_path)),
                "row_count": row_count,
            }
        )
        total_rows += row_count

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_run": generation_run,
        "teacher_model": teacher_model,
        "teacher_provider": "openrouter",
        "token_target": token_target,
        "signals": [dataset["signal"] for dataset in normalized_datasets],
        "datasets": normalized_datasets,
        "total_rows": total_rows,
        "metadata": dict(metadata or {}),
    }

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path
