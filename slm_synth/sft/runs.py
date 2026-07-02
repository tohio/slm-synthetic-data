"""SFT dataset materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.sft.io import write_jsonl
from slm_synth.sft.manifest import write_manifest, write_run_manifest
from slm_synth.sft.seeds import SFT_SEED_FAMILIES, build_seed_rows
from slm_synth.taxonomy.holdouts import HoldoutRegistry


@dataclass(frozen=True)
class SFTSeedRunResult:
    """Result of materializing one SFT seed dataset."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    family: str
    generation_run: str


@dataclass(frozen=True)
class SFTSeedFamilyRunResult:
    """Result of materializing multiple SFT seed families."""

    results: tuple[SFTSeedRunResult, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


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


def materialize_seed_run(
    *,
    families: list[str] | tuple[str, ...] | None,
    count_per_family: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    generation_run: str,
    start_index: int = 1,
    run_manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> SFTSeedFamilyRunResult:
    """Materialize one SFT seed dataset per requested family."""
    resolved_families = resolve_seed_families(families)
    if not isinstance(count_per_family, int) or count_per_family < 1:
        raise ValueError("count_per_family must be a positive integer")

    results = tuple(
        materialize_seed_dataset(
            family=family,
            count=count_per_family,
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            generation_run=generation_run,
            start_index=start_index,
            metadata={
                "seed_run_family_count": len(resolved_families),
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
        )
        for family in resolved_families
    )
    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    write_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        datasets=[
            {
                "family": result.family,
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
            for result in results
        ],
        metadata={
            "family_count": len(resolved_families),
            "count_per_family": count_per_family,
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )

    return SFTSeedFamilyRunResult(
        results=results,
        row_count=sum(result.row_count for result in results),
        families=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_seed_families(families: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve requested SFT seed families, where None or ['all'] means all families."""
    if families is None or tuple(families) == ("all",):
        return tuple(sorted(SFT_SEED_FAMILIES))
    if "all" in families:
        raise ValueError("'all' cannot be combined with explicit SFT seed families")

    resolved: list[str] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("SFT seed family must be a non-empty string")
        normalized = family.strip().lower()
        if normalized not in SFT_SEED_FAMILIES:
            supported = ", ".join(sorted(SFT_SEED_FAMILIES))
            raise ValueError(f"Unsupported SFT seed family '{family}'. Supported families: {supported}")
        if normalized in seen:
            raise ValueError(f"Duplicate SFT seed family: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one SFT seed family is required")
    return tuple(resolved)
