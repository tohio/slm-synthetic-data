"""SFT dataset materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.sft.io import write_jsonl
from slm_synth.sft.manifest import write_manifest
from slm_synth.sft.seeds import build_seed_rows
from slm_synth.taxonomy.holdouts import HoldoutRegistry


@dataclass(frozen=True)
class SFTSeedRunResult:
    """Result of materializing one SFT seed dataset."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    family: str
    generation_run: str


def default_dataset_path(*, output_dir: str | Path, family: str) -> Path:
    """Return the default SFT JSONL path for a seed family."""
    return Path(output_dir) / f"{family}.jsonl"


def default_manifest_path(*, manifest_dir: str | Path, family: str, generation_run: str) -> Path:
    """Return the default local SFT manifest path for a seed run."""
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{family}.{generation_run}.manifest.json"


def materialize_seed_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    generation_run: str,
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> SFTSeedRunResult:
    """Build SFT seed rows and write JSONL plus a local manifest."""
    rows = build_seed_rows(
        family=family,
        count=count,
        start_index=start_index,
        holdout_registry=holdout_registry,
    )

    dataset_path = Path(output_dir) / dataset_filename if dataset_filename else default_dataset_path(
        output_dir=output_dir,
        family=family,
    )
    row_count = write_jsonl(rows, dataset_path)

    manifest_path = Path(manifest_dir) / manifest_filename if manifest_filename else default_manifest_path(
        manifest_dir=manifest_dir,
        family=family,
        generation_run=generation_run,
    )
    write_manifest(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        rows=rows,
        generation_run=generation_run,
        metadata={
            "family": family,
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )

    return SFTSeedRunResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=row_count,
        family=family,
        generation_run=generation_run,
    )
