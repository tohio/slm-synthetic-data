"""Normalize public dataset counts in generation manifests.

The public JSONL files are the source of truth. This module updates only
aggregate run/family manifests; batch manifests and generation accounting stay
untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BATCH_MANIFEST_RE = re.compile(r"\.batch\d+\.")

_KIND_TOTAL_FIELD = {
    "pretrain": "total_records",
    "sft": "total_rows",
    "dpo": "total_pairs",
    "distillation-sft": "total_rows",
    "distillation-dpo": "total_pairs",
}

_KIND_UNIT = {
    "pretrain": "records",
    "sft": "rows",
    "dpo": "pairs",
    "distillation-sft": "rows",
    "distillation-dpo": "pairs",
}

_KIND_METADATA_KEY = {
    "pretrain": "signal",
    "sft": "task_family",
    "dpo": "preference_dimension",
    "distillation-sft": "signal",
    "distillation-dpo": "preference_dimension",
}

_KIND_PUBLISHED_TOTAL = {
    "pretrain": "published_records",
    "sft": "published_rows",
    "dpo": "published_pairs",
    "distillation-sft": "published_rows",
    "distillation-dpo": "published_pairs",
}

_KIND_PUBLISHED_DISTRIBUTION = {
    "pretrain": "published_signal_counts",
    "sft": "published_family_counts",
    "dpo": "published_dimension_counts",
    "distillation-sft": "published_signal_counts",
    "distillation-dpo": "published_dimension_counts",
}


@dataclass(frozen=True)
class PublicDatasetStats:
    total: int
    file_counts: dict[str, int]
    metadata_counts: dict[str, int]


@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    data: dict[str, Any]



def normalize_run(*, kind: str, run_dir: Path) -> list[Path]:
    """Normalize aggregate manifest totals from the final public JSONL files.

    The run directory name is not assumed to equal ``generation_run``. This is
    important for copied publication bundles such as ``distillation_sft/`` that
    retain a production manifest named after the original generation run.
    """

    _validate_kind(kind)
    run_dir = run_dir.resolve()
    manifests_dir = run_dir / "manifests"
    if not manifests_dir.is_dir():
        raise FileNotFoundError(f"manifest directory is missing: {manifests_dir}")

    public_dir = _find_public_dataset_dir(run_dir=run_dir, kind=kind)
    stats = _scan_public_dataset(public_dir=public_dir, kind=kind)
    manifests = _load_aggregate_manifests(manifests_dir=manifests_dir, kind=kind)
    run_manifest = _select_run_manifest(manifests=manifests, kind=kind)
    run_id = _resolve_run_id(
        run_dir=run_dir,
        manifests=manifests,
        run_manifest=run_manifest,
    )

    changed: list[Path] = []
    for record in _manifests_for_run(
        manifests=manifests,
        run_id=run_id,
        run_manifest=run_manifest,
    ):
        family = None if run_manifest and record.path == run_manifest.path else _manifest_family(
            record=record,
            run_id=run_id,
        )
        count = _count_for_manifest(stats=stats, family=family)
        changed.append(
            _normalize_manifest_file(
                kind=kind,
                record=record,
                count=count,
                stats=stats if family is None else None,
            )
        )

    if not changed:
        raise FileNotFoundError(f"no aggregate manifests found under {manifests_dir}")
    return changed


def load_run_manifest(*, run_dir: Path, kind: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Load the aggregate run manifest without relying on the directory name."""

    root = Path(run_dir).resolve()
    manifests_dir = root / "manifests"
    if not manifests_dir.is_dir():
        raise FileNotFoundError(f"manifest directory is missing: {manifests_dir}")

    records = _load_aggregate_manifests(manifests_dir=manifests_dir, kind=kind)
    selected = _select_run_manifest(manifests=records, kind=kind)
    if selected is None:
        names = ", ".join(record.path.name for record in records)
        raise FileNotFoundError(
            f"no aggregate run manifest found under {manifests_dir}; candidates: {names}"
        )
    return selected.path, dict(selected.data)


def public_dataset_stats(*, kind: str, run_dir: Path) -> PublicDatasetStats:
    """Return public counts using the same boundary as manifest normalization."""

    _validate_kind(kind)
    root = Path(run_dir).resolve()
    public_dir = _find_public_dataset_dir(run_dir=root, kind=kind)
    return _scan_public_dataset(public_dir=public_dir, kind=kind)


def _validate_kind(kind: str) -> None:
    if kind not in _KIND_TOTAL_FIELD:
        raise ValueError(f"unsupported generation kind: {kind}")


def _find_public_dataset_dir(*, run_dir: Path, kind: str) -> Path:
    candidates = [run_dir / "datasets", run_dir / "data"]
    if kind == "pretrain":
        candidates.extend([run_dir / "deduped", run_dir / "dataset", run_dir])

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jsonl")):
            return candidate

    raise FileNotFoundError(
        f"public JSONL dataset directory is missing for {kind}: "
        f"checked {', '.join(str(path) for path in candidates)}"
    )


def _scan_public_dataset(*, public_dir: Path, kind: str) -> PublicDatasetStats:
    metadata_key = _KIND_METADATA_KEY[kind]
    file_counts: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    metadata_rows = 0
    missing_metadata_rows = 0

    paths = sorted(public_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no public JSONL files found under {public_dir}")

    total = 0
    for path in paths:
        file_key = path.stem.split(".batch", 1)[0]
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"public row must be an object: {path}:{line_number}")

                total += 1
                file_counts[file_key] += 1
                metadata = row.get("metadata")
                value = metadata.get(metadata_key) if isinstance(metadata, Mapping) else None
                if isinstance(value, str) and value.strip():
                    metadata_rows += 1
                    metadata_counts[value.strip()] += 1
                else:
                    missing_metadata_rows += 1

    if metadata_rows and missing_metadata_rows:
        raise ValueError(
            f"public {kind} rows inconsistently populate metadata.{metadata_key}: "
            f"present={metadata_rows} missing={missing_metadata_rows}"
        )

    return PublicDatasetStats(
        total=total,
        file_counts=dict(sorted(file_counts.items())),
        metadata_counts=dict(sorted(metadata_counts.items())),
    )


def _load_aggregate_manifests(
    *, manifests_dir: Path, kind: str | None
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for path in sorted(manifests_dir.glob("*.manifest.json")):
        if _BATCH_MANIFEST_RE.search(path.name):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON: {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"manifest must contain an object: {path}")
        dataset_type = value.get("dataset_type")
        if kind is not None and dataset_type not in {None, kind}:
            continue
        records.append(ManifestRecord(path=path.resolve(), data=value))

    if not records:
        expected = f" for {kind}" if kind else ""
        raise FileNotFoundError(f"no aggregate manifests found{expected} under {manifests_dir}")
    return records


def _select_run_manifest(
    *, manifests: list[ManifestRecord], kind: str | None
) -> ManifestRecord | None:
    exact: list[ManifestRecord] = []
    structured: list[ManifestRecord] = []

    for record in manifests:
        generation_run = record.data.get("generation_run")
        if (
            isinstance(generation_run, str)
            and generation_run.strip()
            and record.path.name == f"{generation_run.strip()}.manifest.json"
        ):
            exact.append(record)

        datasets = record.data.get("datasets")
        data_entries = record.data.get("data")
        if isinstance(datasets, list) or isinstance(data_entries, list):
            structured.append(record)
        elif (kind == "pretrain" or record.data.get("dataset_type") == "pretrain") and isinstance(
            record.data.get("stages"), Mapping
        ):
            structured.append(record)

    candidates = exact or structured
    unique = {record.path: record for record in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if not unique:
        return None
    names = ", ".join(path.name for path in sorted(unique))
    raise ValueError(f"expected one aggregate run manifest; found {len(unique)}: {names}")


def _resolve_run_id(
    *,
    run_dir: Path,
    manifests: list[ManifestRecord],
    run_manifest: ManifestRecord | None,
) -> str:
    if run_manifest is not None:
        value = run_manifest.data.get("generation_run")
        if isinstance(value, str) and value.strip():
            return value.strip()

    values = {
        value.strip()
        for record in manifests
        if isinstance((value := record.data.get("generation_run")), str) and value.strip()
    }
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1:
        raise ValueError(f"manifests contain multiple generation_run values: {sorted(values)}")
    return run_dir.name


def _manifests_for_run(
    *,
    manifests: list[ManifestRecord],
    run_id: str,
    run_manifest: ManifestRecord | None,
) -> list[ManifestRecord]:
    suffix = f".{run_id}.manifest.json"
    selected: list[ManifestRecord] = []
    for record in manifests:
        value = record.data.get("generation_run")
        if isinstance(value, str) and value.strip() and value.strip() != run_id:
            continue
        if run_manifest is not None and record.path == run_manifest.path:
            selected.append(record)
        elif record.path.name.endswith(suffix):
            selected.append(record)
        elif run_manifest is None and len(manifests) == 1:
            selected.append(record)
    return selected


def _manifest_family(*, record: ManifestRecord, run_id: str) -> str | None:
    for field in ("signal", "family", "preference_dimension"):
        value = record.data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    suffix = f".{run_id}.manifest.json"
    if record.path.name.endswith(suffix):
        family = record.path.name[: -len(suffix)]
        return family or None
    return None


def _distribution_for_stats(stats: PublicDatasetStats) -> dict[str, int]:
    return stats.metadata_counts or stats.file_counts


def _count_for_manifest(*, stats: PublicDatasetStats, family: str | None) -> int:
    if family is None:
        return stats.total
    if family in stats.file_counts:
        return stats.file_counts[family]
    distribution = _distribution_for_stats(stats)
    if family in distribution:
        return distribution[family]
    raise FileNotFoundError(
        f"public family dataset/count is missing for {family!r}; "
        f"files={sorted(stats.file_counts)} metadata_groups={sorted(stats.metadata_counts)}"
    )


def _normalize_manifest_file(
    *,
    kind: str,
    record: ManifestRecord,
    count: int,
    stats: PublicDatasetStats | None,
) -> Path:
    data = dict(record.data)
    _set_canonical_total_fields(kind=kind, manifest=data, count=count)
    _validate_accepted_target(
        kind=kind,
        manifest_path=record.path,
        manifest=data,
        count=count,
    )

    if stats is not None:
        distribution = _distribution_for_stats(stats)
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata
        metadata[_KIND_PUBLISHED_TOTAL[kind]] = stats.total
        metadata[_KIND_PUBLISHED_DISTRIBUTION[kind]] = dict(distribution)
        _normalize_dataset_entries(
            manifest=data,
            stats=stats,
            distribution=distribution,
        )

    record.path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record.path


def _normalize_dataset_entries(
    *,
    manifest: dict[str, Any],
    stats: PublicDatasetStats,
    distribution: Mapping[str, int],
) -> None:
    for field in ("datasets", "data"):
        entries = manifest.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = next(
                (
                    value.strip()
                    for name in ("signal", "family", "preference_dimension")
                    if isinstance((value := entry.get(name)), str) and value.strip()
                ),
                None,
            )
            if key in stats.file_counts:
                entry["row_count"] = stats.file_counts[key]
            elif key in distribution:
                entry["row_count"] = int(distribution[key])
            elif len(entries) == 1:
                entry["row_count"] = stats.total


def _set_canonical_total_fields(*, kind: str, manifest: dict[str, Any], count: int) -> None:
    if kind == "pretrain":
        manifest["total_records"] = count
        manifest["total_rows"] = count
        manifest["total_pairs"] = None
        return

    total_field = _KIND_TOTAL_FIELD[kind]
    for field in ("total_rows", "total_pairs", "total_records"):
        manifest[field] = count if field == total_field else None


def _validate_accepted_target(
    *,
    kind: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    count: int,
) -> None:
    metadata = manifest.get("metadata")
    accepted_target = metadata.get("accepted_target") if isinstance(metadata, Mapping) else None
    if not isinstance(accepted_target, Mapping):
        return

    unit = accepted_target.get("unit")
    expected_unit = _KIND_UNIT[kind]
    if unit != expected_unit:
        return

    accepted = accepted_target.get("accepted")
    if accepted is None:
        return
    if int(accepted) != count:
        raise ValueError(
            f"{manifest_path} accepted_target.accepted={accepted} does not match "
            f"public {expected_unit} count={count}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize generation manifest total fields.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize = subparsers.add_parser("normalize", help="Normalize one run directory.")
    normalize.add_argument("--kind", required=True, choices=sorted(_KIND_TOTAL_FIELD))
    normalize.add_argument("--run-dir", required=True, type=Path)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "normalize":
        changed = normalize_run(kind=args.kind, run_dir=args.run_dir)
        for path in changed:
            print(f"[manifest_totals] normalized {path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(cli())
