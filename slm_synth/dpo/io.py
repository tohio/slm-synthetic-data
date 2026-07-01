"""JSONL writing helpers for synthetic DPO datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Write validated DPO rows to JSONL and return the row count."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validated = validate_dpo_row(row)
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate DPO rows from JSONL."""
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {input_path} at line {line_number}: {exc}") from exc
        rows.append(validate_dpo_row(value))
    return rows
