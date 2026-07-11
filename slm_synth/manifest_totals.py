from __future__ import annotations

import argparse
import json
import re
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


def normalize_run(*, kind: str, run_dir: Path) -> list[Path]:
    """Normalize aggregate manifest totals for one generation run.

    Batch manifests are intentionally ignored. Only these manifest types are
    normalized:
      - manifests/<run>.manifest.json
      - manifests/<family>.<run>.manifest.json
    """

    if kind not in _KIND_TOTAL_FIELD:
        raise ValueError(f"unsupported generation kind: {kind}")

    run_dir = run_dir.resolve()
    manifests_dir = run_dir / "manifests"
    if not manifests_dir.exists():
        raise FileNotFoundError(f"manifest directory is missing: {manifests_dir}")

    public_dir = _find_public_dataset_dir(run_dir=run_dir, kind=kind)
    run_id = run_dir.name

    changed: list[Path] = []
    for manifest_path in _aggregate_manifests(manifests_dir=manifests_dir, run_id=run_id):
        family = _manifest_family(manifest_path=manifest_path, run_id=run_id)
        count = _count_public_rows(public_dir=public_dir, family=family)
        changed.append(_normalize_manifest_file(kind=kind, manifest_path=manifest_path, count=count))

    return changed


def _find_public_dataset_dir(*, run_dir: Path, kind: str) -> Path:
    candidates = [run_dir / "datasets"]

    if kind == "pretrain":
        candidates.extend(
            [
                run_dir / "data",
                run_dir / "deduped",
                run_dir / "dataset",
                run_dir,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.jsonl")):
            return candidate

    raise FileNotFoundError(
        f"public JSONL dataset directory is missing for {kind}: "
        f"checked {', '.join(str(path) for path in candidates)}"
    )


def _aggregate_manifests(*, manifests_dir: Path, run_id: str) -> list[Path]:
    run_manifest = manifests_dir / f"{run_id}.manifest.json"
    manifests: list[Path] = []

    if run_manifest.exists():
        manifests.append(run_manifest)

    suffix = f".{run_id}.manifest.json"
    for path in sorted(manifests_dir.glob("*.manifest.json")):
        if path == run_manifest:
            continue
        if _BATCH_MANIFEST_RE.search(path.name):
            continue
        if path.name.endswith(suffix):
            manifests.append(path)

    if not manifests:
        raise FileNotFoundError(f"no aggregate manifests found under {manifests_dir}")

    return manifests


def _manifest_family(*, manifest_path: Path, run_id: str) -> str | None:
    if manifest_path.name == f"{run_id}.manifest.json":
        return None

    suffix = f".{run_id}.manifest.json"
    if manifest_path.name.endswith(suffix):
        family = manifest_path.name[: -len(suffix)]
        return family or None

    return None


def _count_public_rows(*, public_dir: Path, family: str | None) -> int:
    if family:
        paths = [public_dir / f"{family}.jsonl"]
        if not paths[0].exists():
            raise FileNotFoundError(f"public family dataset is missing: {paths[0]}")
    else:
        paths = sorted(public_dir.glob("*.jsonl"))

    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _normalize_manifest_file(*, kind: str, manifest_path: Path, count: int) -> Path:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    _set_canonical_total_fields(kind=kind, manifest=data, count=count)

    _validate_accepted_target(kind=kind, manifest_path=manifest_path, manifest=data, count=count)

    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path



def _set_canonical_total_fields(*, kind: str, manifest: dict[str, Any], count: int) -> None:
    # Set only the canonical total field for this generation kind.
    #
    # Pretrain keeps total_rows as a compatibility alias. Pair-based datasets do
    # not keep total_rows because their public unit is pairs, not rows.

    if kind == "pretrain":
        manifest["total_records"] = count
        manifest["total_rows"] = count
        manifest["total_pairs"] = None
        return

    total_field = _KIND_TOTAL_FIELD[kind]
    for field in ("total_rows", "total_pairs", "total_records"):
        manifest[field] = count if field == total_field else None

def _validate_accepted_target(*, kind: str, manifest_path: Path, manifest: dict[str, Any], count: int) -> None:
    accepted_target = manifest.get("metadata", {}).get("accepted_target")
    if not isinstance(accepted_target, dict):
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
