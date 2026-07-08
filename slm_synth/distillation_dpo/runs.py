"""Run helpers for isolated distillation-DPO artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from slm_synth.distillation_dpo.io import write_family_dataset, write_manifest, write_run_manifest
from slm_synth.distillation_dpo.seeds import DISTILLATION_DPO_FAMILIES, build_seed_rows, validate_family


@dataclass(frozen=True)
class DistillationDPOFamilyResult:
    family: str
    dataset_path: Path
    manifest_path: Path
    row_count: int


@dataclass(frozen=True)
class DistillationDPORunResult:
    generation_run: str
    results: tuple[DistillationDPOFamilyResult, ...]
    manifest_path: Path

    @property
    def row_count(self) -> int:
        return sum(result.row_count for result in self.results)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(result.family for result in self.results)


def default_manifest_path(*, manifest_dir: str | Path, family: str, generation_run: str) -> Path:
    normalized_family = validate_family(family)
    return Path(manifest_dir) / f"{normalized_family}.{generation_run}.manifest.json"


def default_run_manifest_path(*, manifest_dir: str | Path, generation_run: str) -> Path:
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{generation_run}.manifest.json"


def resolve_families(families: Sequence[str] | None = None) -> tuple[str, ...]:
    if families is None or not families or "all" in families:
        if families and len(families) > 1:
            raise ValueError("'all' cannot be combined with explicit distillation-DPO families")
        return tuple(sorted(DISTILLATION_DPO_FAMILIES))

    normalized = tuple(validate_family(family) for family in families)
    duplicates = sorted({family for family in normalized if normalized.count(family) > 1})
    if duplicates:
        raise ValueError(f"duplicate distillation-DPO family/families requested: {duplicates}")
    return normalized


def materialize_seed_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    token_target: str | int | None = None,
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
) -> DistillationDPOFamilyResult:
    """Materialize one smoke/control distillation-DPO family dataset."""
    normalized_family = validate_family(family)
    rows = build_seed_rows(family=normalized_family, count=count, start_index=start_index)
    dataset_path = write_family_dataset(
        family=normalized_family,
        rows=rows,
        output_dir=output_dir,
        filename=dataset_filename,
    )
    manifest_path = (
        Path(manifest_dir) / manifest_filename
        if manifest_filename is not None
        else default_manifest_path(manifest_dir=manifest_dir, family=normalized_family, generation_run=generation_run)
    )
    write_manifest(
        manifest_path=manifest_path,
        family=normalized_family,
        dataset_path=dataset_path,
        row_count=len(rows),
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        metadata={"generation_mode": "seed_controlled_weak"},
    )
    return DistillationDPOFamilyResult(
        family=normalized_family,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(rows),
    )


def materialize_seed_run(
    *,
    families: Sequence[str] | None,
    count_per_family: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    token_target: str | int | None = None,
    start_index: int = 1,
    run_manifest_filename: str | None = None,
) -> DistillationDPORunResult:
    """Materialize a deterministic distillation-DPO smoke/control run."""
    normalized_families = resolve_families(families)
    results: list[DistillationDPOFamilyResult] = []
    for family in normalized_families:
        results.append(
            materialize_seed_dataset(
                family=family,
                count=count_per_family,
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                teacher_model=teacher_model,
                teacher_provider=teacher_provider,
                generation_run=generation_run,
                token_target=token_target,
                start_index=start_index,
            )
        )

    manifest_path = (
        Path(manifest_dir) / run_manifest_filename
        if run_manifest_filename is not None
        else default_run_manifest_path(manifest_dir=manifest_dir, generation_run=generation_run)
    )
    write_run_manifest(
        manifest_path=manifest_path,
        generation_run=generation_run,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=token_target,
        datasets=(
            {
                "family": result.family,
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
            for result in results
        ),
        metadata={
            "generation_mode": "seed_controlled_weak_run",
            "count_per_family": count_per_family,
        },
    )
    return DistillationDPORunResult(
        generation_run=generation_run,
        results=tuple(results),
        manifest_path=manifest_path,
    )
