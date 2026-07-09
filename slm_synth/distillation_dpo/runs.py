"""Run helpers for isolated distillation-DPO artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata, raise_for_underfilled_manifest
from slm_synth.distillation_dpo.io import write_family_dataset, write_manifest, write_run_manifest
from slm_synth.distillation_dpo.pair_quality import (
    PairQualitySummary,
    aggregate_rejection_reasons,
    filter_pairs_by_quality,
)
from slm_synth.distillation_dpo.seeds import DISTILLATION_DPO_FAMILIES, build_seed_rows, validate_family
from slm_synth.distillation_dpo.spec_builders import build_production_rows


@dataclass(frozen=True)
class DistillationDPOFamilyResult:
    family: str
    dataset_path: Path
    manifest_path: Path
    row_count: int
    planned_pairs: int
    accepted_pairs: int
    rejected_pairs: int
    rejection_reasons: dict[str, int]
    max_backfill_rounds: int = 0
    backfill_rounds: int = 0


@dataclass(frozen=True)
class DistillationDPORunResult:
    generation_run: str
    results: tuple[DistillationDPOFamilyResult, ...]
    manifest_path: Path
    target_pairs: int | None = None
    planned_pairs: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    rejection_reasons: dict[str, int] | None = None

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


def normalize_family_pair_counts(
    *,
    families: Sequence[str] | None,
    target_pairs: int,
) -> dict[str, int]:
    """Split a run-level target pair count deterministically across families."""
    normalized_families = resolve_families(families)
    _validate_positive_int(target_pairs, "target_pairs")
    if target_pairs < len(normalized_families):
        raise ValueError("target_pairs must be at least the number of requested distillation-DPO families")

    base = target_pairs // len(normalized_families)
    remainder = target_pairs % len(normalized_families)
    counts: dict[str, int] = {}
    for index, family in enumerate(normalized_families):
        counts[family] = base + (1 if index < remainder else 0)
    return counts


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
    max_backfill_rounds: int = 2,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    """Materialize one smoke/control distillation-DPO family dataset."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    rows, accepted_rows, quality, backfill_rounds = _build_rows_until_target(
        family=normalized_family,
        target_pairs=count,
        start_index=start_index,
        max_backfill_rounds=max_backfill_rounds,
        builder=lambda *, count, start_index: build_seed_rows(
            family=normalized_family, count=count, start_index=start_index
        ),
    )
    return _materialize_family_dataset(
        family=normalized_family,
        accepted_rows=accepted_rows,
        quality=quality,
        target_pairs=count,
        planned_pairs=len(rows),
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        dataset_filename=dataset_filename,
        manifest_filename=manifest_filename,
        metadata={"generation_mode": "seed_controlled_weak"},
        raise_on_underfill=raise_on_underfill,
    )


def materialize_production_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    """Materialize one production distillation-DPO family dataset."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    rows, accepted_rows, quality, backfill_rounds = _build_rows_until_target(
        family=normalized_family,
        target_pairs=count,
        start_index=start_index,
        max_backfill_rounds=max_backfill_rounds,
        builder=lambda *, count, start_index: build_production_rows(
            family=normalized_family, count=count, start_index=start_index
        ),
    )
    return _materialize_family_dataset(
        family=normalized_family,
        accepted_rows=accepted_rows,
        quality=quality,
        target_pairs=count,
        planned_pairs=len(rows),
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=None,
        dataset_filename=dataset_filename,
        manifest_filename=manifest_filename,
        metadata={"generation_mode": "production_controlled_weak"},
        raise_on_underfill=raise_on_underfill,
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
    max_backfill_rounds: int = 2,
) -> DistillationDPORunResult:
    """Materialize a deterministic distillation-DPO smoke/control run."""
    _validate_positive_int(count_per_family, "count_per_family")
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
                max_backfill_rounds=max_backfill_rounds,
                raise_on_underfill=False,
            )
        )

    return _write_run_result(
        generation_run=generation_run,
        results=results,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=token_target,
        target_pairs=len(normalized_families) * count_per_family,
        run_manifest_filename=run_manifest_filename,
        metadata={
            "generation_mode": "seed_controlled_weak_run",
            "count_per_family": count_per_family,
            "max_backfill_rounds": max_backfill_rounds,
        },
    )


def materialize_production_run(
    *,
    families: Sequence[str] | None,
    target_pairs: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    start_index: int = 1,
    run_manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
) -> DistillationDPORunResult:
    """Materialize a deterministic production distillation-DPO run."""
    pair_counts = normalize_family_pair_counts(families=families, target_pairs=target_pairs)
    results: list[DistillationDPOFamilyResult] = []
    for family, count in pair_counts.items():
        results.append(
            materialize_production_dataset(
                family=family,
                count=count,
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                teacher_model=teacher_model,
                teacher_provider=teacher_provider,
                generation_run=generation_run,
                start_index=start_index,
                max_backfill_rounds=max_backfill_rounds,
                raise_on_underfill=False,
            )
        )

    return _write_run_result(
        generation_run=generation_run,
        results=results,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=None,
        target_pairs=target_pairs,
        run_manifest_filename=run_manifest_filename,
        metadata={
            "generation_mode": "production_controlled_weak_run",
            "target_pairs": target_pairs,
            "pairs_per_family": pair_counts,
            "max_backfill_rounds": max_backfill_rounds,
        },
    )


def _build_rows_until_target(
    *,
    family: str,
    target_pairs: int,
    start_index: int,
    max_backfill_rounds: int,
    builder: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PairQualitySummary, int]:
    _validate_positive_int(target_pairs, "target_pairs")
    _validate_positive_int(start_index, "start_index")
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")

    attempted_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    quality = PairQualitySummary(checked_pairs=0, accepted_pairs=0, rejected_pairs=0, rejection_reasons={})
    next_start_index = start_index
    backfill_rounds = 0

    while len(accepted_rows) < target_pairs:
        if attempted_rows and backfill_rounds >= max_backfill_rounds:
            break
        remaining = target_pairs - len(accepted_rows)
        if attempted_rows:
            backfill_rounds += 1
        new_rows = builder(count=remaining, start_index=next_start_index)
        attempted_rows.extend(new_rows)
        next_start_index += remaining
        accepted_rows, quality = filter_pairs_by_quality(family=family, rows=attempted_rows)

    if len(accepted_rows) > target_pairs:
        accepted_rows = accepted_rows[:target_pairs]
        quality = PairQualitySummary(
            checked_pairs=quality.checked_pairs,
            accepted_pairs=len(accepted_rows),
            rejected_pairs=max(quality.checked_pairs - len(accepted_rows), 0),
            rejection_reasons=quality.rejection_reasons,
        )
    return attempted_rows, accepted_rows, quality, backfill_rounds


def _materialize_family_dataset(
    *,
    family: str,
    accepted_rows: Sequence[Mapping[str, Any]],
    quality: PairQualitySummary,
    target_pairs: int,
    planned_pairs: int,
    max_backfill_rounds: int,
    backfill_rounds: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    token_target: str | int | None,
    metadata: Mapping[str, Any],
    dataset_filename: str | None,
    manifest_filename: str | None,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    dataset_path = write_family_dataset(
        family=family,
        rows=accepted_rows,
        output_dir=output_dir,
        filename=dataset_filename,
    )
    manifest_path = (
        Path(manifest_dir) / manifest_filename
        if manifest_filename is not None
        else default_manifest_path(manifest_dir=manifest_dir, family=family, generation_run=generation_run)
    )
    manifest_metadata = _planning_metadata(
        base_metadata=metadata,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        quality=quality,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
    )
    write_manifest(
        manifest_path=manifest_path,
        family=family,
        dataset_path=dataset_path,
        row_count=len(accepted_rows),
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        metadata=manifest_metadata,
    )
    if raise_on_underfill:
        raise_for_underfilled_manifest(manifest_path, artifact_name=f"distillation-dpo family {family}")
    return DistillationDPOFamilyResult(
        family=family,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(accepted_rows),
        planned_pairs=planned_pairs,
        accepted_pairs=quality.accepted_pairs,
        rejected_pairs=quality.rejected_pairs,
        rejection_reasons=quality.rejection_reasons,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
    )


def _write_run_result(
    *,
    generation_run: str,
    results: Sequence[DistillationDPOFamilyResult],
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    token_target: str | int | None,
    target_pairs: int,
    run_manifest_filename: str | None,
    metadata: Mapping[str, Any],
) -> DistillationDPORunResult:
    planned_pairs = sum(result.planned_pairs for result in results)
    accepted_pairs = sum(result.accepted_pairs for result in results)
    rejected_pairs = sum(result.rejected_pairs for result in results)
    rejection_reasons = _merge_rejection_reasons(result.rejection_reasons for result in results)
    quality_summary = PairQualitySummary(
        checked_pairs=planned_pairs,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
    )
    manifest_metadata = _planning_metadata(
        base_metadata=metadata,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        quality=quality_summary,
        max_backfill_rounds=max(_metadata_max_backfill_rounds(results), int(metadata.get("max_backfill_rounds", 0)) if isinstance(metadata.get("max_backfill_rounds"), int) else 0),
        backfill_rounds=max(_metadata_backfill_rounds(results), 0),
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
        metadata=manifest_metadata,
    )
    raise_for_underfilled_manifest(manifest_path, artifact_name="distillation-dpo")
    return DistillationDPORunResult(
        generation_run=generation_run,
        results=tuple(results),
        manifest_path=manifest_path,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
    )


def _planning_metadata(
    *,
    base_metadata: Mapping[str, Any],
    target_pairs: int,
    planned_pairs: int,
    quality: PairQualitySummary,
    max_backfill_rounds: int,
    backfill_rounds: int,
) -> dict[str, Any]:
    metadata = dict(base_metadata)
    metadata.update(
        {
            "source_contract": {
                "chosen_source": "teacher",
                "rejected_source": "controlled_weak",
                "target_consumer": "slm-distillation",
            },
            "target_pairs": target_pairs,
            "planned_pairs": planned_pairs,
            "accepted_pairs": quality.accepted_pairs,
            "rejected_pairs": quality.rejected_pairs,
            "rejection_reasons": dict(sorted(quality.rejection_reasons.items())),
            "pair_quality": quality.to_dict(),
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds,
            **accepted_target_metadata(
                unit="pairs",
                target_count=target_pairs,
                accepted_count=quality.accepted_pairs,
                attempted_count=planned_pairs,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=backfill_rounds,
            ),
        }
    )
    return metadata


def _metadata_max_backfill_rounds(results: Sequence[DistillationDPOFamilyResult]) -> int:
    return max((result.max_backfill_rounds for result in results), default=0)


def _metadata_backfill_rounds(results: Sequence[DistillationDPOFamilyResult]) -> int:
    return max((result.backfill_rounds for result in results), default=0)


def _merge_rejection_reasons(summaries: Sequence[Mapping[str, int]] | Any) -> dict[str, int]:
    if not isinstance(summaries, Sequence):
        return aggregate_rejection_reasons([])
    counter: Counter[str] = Counter()
    for summary in summaries:
        if isinstance(summary, Mapping):
            for reason, count in summary.items():
                if isinstance(reason, str) and isinstance(count, int) and count > 0:
                    counter[reason] += count
    return dict(sorted(counter.items()))


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
