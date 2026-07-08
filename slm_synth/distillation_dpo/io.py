"""JSONL and manifest writing helpers for distillation-DPO artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row
from slm_synth.distillation_dpo.seeds import validate_family

DATASET_TYPE = "distillation-dpo"
CHOSEN_SOURCE = "teacher"
REJECTED_SOURCE = "student_or_controlled_weak"
TARGET_CONSUMER = "slm-distillation"


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Write validated public distillation-DPO rows and return row count."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validated = validate_distillation_dpo_row(row)
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate public distillation-DPO rows from JSONL."""
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {input_path} at line {line_number}: {exc}") from exc
        rows.append(validate_distillation_dpo_row(value))
    return rows


def write_family_dataset(
    *,
    family: str,
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Write one family-specific public distillation-DPO dataset."""
    normalized_family = validate_family(family)
    output_path = Path(output_dir) / (filename or f"{normalized_family}.jsonl")
    write_jsonl(rows, output_path)
    return output_path


def write_manifest(
    *,
    manifest_path: str | Path,
    family: str,
    dataset_path: str | Path,
    row_count: int,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    token_target: str | int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local-only manifest for one distillation-DPO family."""
    normalized_family = validate_family(family)
    teacher_model = _require_non_empty_string(teacher_model, "teacher_model")
    if teacher_provider.strip().lower() != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")
    generation_run = _require_non_empty_string(generation_run, "generation_run")
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": DATASET_TYPE,
        "family": normalized_family,
        "dataset_path": str(Path(dataset_path)),
        "row_count": row_count,
        "teacher_model": teacher_model,
        "teacher_provider": "openrouter",
        "generation_run": generation_run,
        "token_target": token_target,
        "chosen_source": CHOSEN_SOURCE,
        "rejected_source": REJECTED_SOURCE,
        "target_consumer": TARGET_CONSUMER,
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
    """Write a local-only manifest summarizing a distillation-DPO run."""
    generation_run = _require_non_empty_string(generation_run, "generation_run")
    teacher_model = _require_non_empty_string(teacher_model, "teacher_model")
    if teacher_provider.strip().lower() != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")

    normalized_datasets: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    total_rows = 0
    for dataset in datasets:
        family = validate_family(dataset.get("family"))
        if family in seen_families:
            raise ValueError(f"duplicate family in run manifest datasets: {family}")
        seen_families.add(family)

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
                "family": family,
                "dataset_path": str(Path(dataset_path)),
                "manifest_path": str(Path(manifest_item_path)),
                "row_count": row_count,
            }
        )
        total_rows += row_count

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": DATASET_TYPE,
        "generation_run": generation_run,
        "teacher_model": teacher_model,
        "teacher_provider": "openrouter",
        "token_target": token_target,
        "chosen_source": CHOSEN_SOURCE,
        "rejected_source": REJECTED_SOURCE,
        "target_consumer": TARGET_CONSUMER,
        "families": [dataset["family"] for dataset in normalized_datasets],
        "datasets": normalized_datasets,
        "total_rows": total_rows,
        "metadata": dict(metadata or {}),
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
