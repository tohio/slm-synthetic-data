import json

import pytest

from slm_synth.accepted_target import (
    UnderfilledRunError,
    accepted_target_metadata,
    discover_run_manifest,
    raise_for_underfilled_manifest,
    require_publish_ready_manifest,
)


def test_accepted_target_metadata_marks_complete_run_publish_ready():
    metadata = accepted_target_metadata(
        unit="rows",
        target_count=10,
        accepted_count=10,
        attempted_count=12,
        max_backfill_rounds=2,
        backfill_rounds=1,
    )

    assert metadata["generation_status"] == "complete"
    assert metadata["publish_ready"] is True
    assert metadata["remaining_rows"] == 0
    assert metadata["accepted_target"] == {
        "unit": "rows",
        "target": 10,
        "accepted": 10,
        "attempted": 12,
        "remaining": 0,
        "status": "complete",
        "publish_ready": True,
        "max_backfill_rounds": 2,
        "backfill_rounds": 1,
        "backfill_budget_exhausted": False,
    }


def test_accepted_target_metadata_marks_underfilled_run_not_publish_ready():
    metadata = accepted_target_metadata(
        unit="pairs",
        target_count=10,
        accepted_count=8,
        attempted_count=12,
        max_backfill_rounds=2,
        backfill_rounds=2,
    )

    assert metadata["generation_status"] == "underfilled"
    assert metadata["publish_ready"] is False
    assert metadata["remaining_pairs"] == 2
    assert metadata["accepted_target"]["remaining"] == 2
    assert metadata["accepted_target"]["backfill_budget_exhausted"] is True
    assert metadata["failure_status"] == "failed"
    assert metadata["failure_reason"] == "accepted_target_underfilled"
    assert metadata["run_failed"] is True


def test_raise_for_underfilled_manifest_raises_after_manifest_is_written(tmp_path):
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": accepted_target_metadata(
                    unit="rows",
                    target_count=3,
                    accepted_count=2,
                    attempted_count=4,
                    max_backfill_rounds=1,
                    backfill_rounds=1,
                )
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnderfilledRunError, match="underfilled.*remaining=1.*accepted=2 target=3"):
        raise_for_underfilled_manifest(manifest_path, artifact_name="SFT")


def test_require_publish_ready_manifest_rejects_underfilled_run(tmp_path):
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": accepted_target_metadata(
                    unit="rows",
                    target_count=2,
                    accepted_count=1,
                    attempted_count=2,
                )
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="underfilled.*remaining=1"):
        require_publish_ready_manifest(manifest_path, artifact_name="SFT")


def test_require_publish_ready_manifest_allows_complete_run(tmp_path):
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": accepted_target_metadata(
                    unit="rows",
                    target_count=2,
                    accepted_count=2,
                    attempted_count=2,
                )
            }
        ),
        encoding="utf-8",
    )

    require_publish_ready_manifest(manifest_path, artifact_name="SFT")


def test_discover_run_manifest_selects_public_run_manifest(tmp_path):
    run_dir = tmp_path / "run"
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    run_manifest = manifest_dir / "run.manifest.json"
    run_manifest.write_text(json.dumps({"dataset_type": "sft", "datasets": []}), encoding="utf-8")
    (manifest_dir / "basic_arithmetic_qa.batch000001.run.manifest.json").write_text(
        json.dumps({"dataset_type": "sft", "batch_number": 1}), encoding="utf-8"
    )

    assert discover_run_manifest(run_dir, dataset_type="sft") == run_manifest
