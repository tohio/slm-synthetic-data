"""Local manifest helpers for synthetic SFT datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.sft.schema import validate_sft_row


def build_manifest_payload(
    *,
    dataset_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    generation_run: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a local SFT dataset manifest from validated row metadata."""
    run = _require_non_empty_string(generation_run, "generation_run")
    validated_rows = [validate_sft_row(row) for row in rows]

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "sft",
        "dataset_path": str(Path(dataset_path)),
        "row_count": len(validated_rows),
        "generation_run": run,
        "categories": _count_metadata(validated_rows, "category"),
        "eval_families": _count_metadata(validated_rows, "eval_family"),
        "template_families": _count_metadata(validated_rows, "template_family"),
        "difficulty_counts": _count_metadata(validated_rows, "difficulty", stringify_keys=True),
        "metadata": dict(metadata or {}),
    }


def write_manifest(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    generation_run: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local SFT manifest and return its path."""
    payload = build_manifest_payload(
        dataset_path=dataset_path,
        rows=rows,
        generation_run=generation_run,
        metadata=metadata,
    )

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_run_manifest(
    *,
    manifest_path: str | Path,
    generation_run: str,
    datasets: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a local manifest summarizing one multi-family SFT run."""
    run = _require_non_empty_string(generation_run, "generation_run")

    normalized_datasets: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    total_rows = 0
    for dataset in datasets:
        family = _require_non_empty_string(dataset.get("family"), "family")
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
        "dataset_type": "sft",
        "generation_run": run,
        "families": [dataset["family"] for dataset in normalized_datasets],
        "datasets": normalized_datasets,
        "total_rows": total_rows,
        "metadata": dict(metadata or {}),
    }

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


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


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
