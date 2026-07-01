"""Local manifest helpers for synthetic DPO datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row


def build_manifest_payload(
    *,
    dataset_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    generation_run: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a local DPO dataset manifest from validated row metadata."""
    run = _require_non_empty_string(generation_run, "generation_run")
    validated_rows = [validate_dpo_row(row) for row in rows]

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "dpo",
        "dataset_path": str(Path(dataset_path)),
        "row_count": len(validated_rows),
        "generation_run": run,
        "categories": _count_metadata(validated_rows, "category"),
        "eval_families": _count_metadata(validated_rows, "eval_family"),
        "template_families": _count_metadata(validated_rows, "template_family"),
        "difficulty_counts": _count_metadata(validated_rows, "difficulty", stringify_keys=True),
        "failure_modes": _count_metadata(validated_rows, "failure_mode"),
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
    """Write a local DPO manifest and return its path."""
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
