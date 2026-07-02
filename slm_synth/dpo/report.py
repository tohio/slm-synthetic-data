"""Coverage reporting helpers for synthetic DPO datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.dpo.io import read_jsonl


def build_coverage_report(paths: list[str | Path]) -> dict[str, Any]:
    """Build an aggregate metadata coverage report for DPO JSONL files."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}

    for path in dataset_paths:
        file_rows = read_jsonl(path)
        file_counts[str(path)] = len(file_rows)
        rows.extend(file_rows)

    return {
        "dataset_type": "dpo",
        "row_count": len(rows),
        "files": file_counts,
        "categories": _count_metadata(rows, "category"),
        "eval_families": _count_metadata(rows, "eval_family"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "failure_modes": _count_metadata(rows, "failure_mode"),
    }


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
