from __future__ import annotations

import json
from pathlib import Path

from slm_synth.manifest_totals import normalize_run


def _write_jsonl(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(json.dumps({"id": f"{path.stem}-{index}"}) + "\n")


def _write_manifest(path: Path, *, unit: str | None = None, accepted: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if unit is not None and accepted is not None:
        metadata["accepted_target"] = {
            "unit": unit,
            "target": accepted,
            "accepted": accepted,
            "attempted": accepted,
            "remaining": 0,
            "status": "complete",
            "publish_ready": True,
        }
    path.write_text(
        json.dumps(
            {
                "generation_run": "run-001",
                "total_rows": None,
                "total_pairs": None,
                "total_records": None,
                "metadata": metadata,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_normalize_sft_run_and_family_manifests_ignores_batch_manifests(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    _write_jsonl(run_dir / "datasets" / "alpha.jsonl", 2)
    _write_jsonl(run_dir / "datasets" / "beta.jsonl", 1)

    run_manifest = run_dir / "manifests" / "run-001.manifest.json"
    family_manifest = run_dir / "manifests" / "alpha.run-001.manifest.json"
    batch_manifest = run_dir / "manifests" / "alpha.batch000001.run-001.manifest.json"

    _write_manifest(run_manifest, unit="rows", accepted=3)
    _write_manifest(family_manifest, unit="rows", accepted=2)
    _write_manifest(batch_manifest)

    changed = normalize_run(kind="sft", run_dir=run_dir)

    assert run_manifest.resolve() in changed
    assert family_manifest.resolve() in changed
    assert batch_manifest.resolve() not in changed

    assert json.loads(run_manifest.read_text())["total_rows"] == 3
    assert json.loads(family_manifest.read_text())["total_rows"] == 2
    assert json.loads(batch_manifest.read_text())["total_rows"] is None


def test_normalize_dpo_sets_total_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    _write_jsonl(run_dir / "datasets" / "alpha.jsonl", 4)

    run_manifest = run_dir / "manifests" / "run-001.manifest.json"
    _write_manifest(run_manifest, unit="pairs", accepted=4)

    normalize_run(kind="dpo", run_dir=run_dir)

    data = json.loads(run_manifest.read_text())
    assert data["total_pairs"] == 4
    assert data["total_rows"] is None


def test_normalize_distillation_dpo_family_sets_total_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    _write_jsonl(run_dir / "datasets" / "teacher_response_preference.jsonl", 5)

    family_manifest = run_dir / "manifests" / "teacher_response_preference.run-001.manifest.json"
    _write_manifest(family_manifest, unit="pairs", accepted=5)

    normalize_run(kind="distillation-dpo", run_dir=run_dir)

    assert json.loads(family_manifest.read_text())["total_pairs"] == 5


def test_normalize_pretrain_sets_total_records_and_compat_total_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    _write_jsonl(run_dir / "datasets" / "pretrain.jsonl", 3)

    run_manifest = run_dir / "manifests" / "run-001.manifest.json"
    _write_manifest(run_manifest, unit="records", accepted=3)

    normalize_run(kind="pretrain", run_dir=run_dir)

    data = json.loads(run_manifest.read_text())
    assert data["total_records"] == 3
    assert data["total_rows"] == 3


def test_normalize_fails_when_accepted_target_mismatches_public_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    _write_jsonl(run_dir / "datasets" / "alpha.jsonl", 1)

    run_manifest = run_dir / "manifests" / "run-001.manifest.json"
    _write_manifest(run_manifest, unit="rows", accepted=2)

    try:
        normalize_run(kind="sft", run_dir=run_dir)
    except ValueError as exc:
        assert "accepted_target.accepted=2" in str(exc)
        assert "count=1" in str(exc)
    else:
        raise AssertionError("expected accepted target mismatch to fail")
