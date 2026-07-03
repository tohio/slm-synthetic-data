import json

from slm_synth.pretrain.artifacts import ArithmeticArtifactFactory
from slm_synth.pretrain.grounded import GroundedBatchStore
from slm_synth.pretrain.report_artifacts import scan_signal
from slm_synth.pretrain.report_lengths import estimated_tokens


def test_artifact_report_reads_persisted_grounded_manifests(tmp_path):
    artifacts = ArithmeticArtifactFactory().build_batch(0, 2)
    records = [{"type": "arithmetic", "question": "Q", "steps": ["S"], "answer": "1"}] * 2
    store = GroundedBatchStore(tmp_path, "arithmetic")
    store.write_completed(batch_id=0, artifacts=artifacts, records=records)
    report = scan_signal(tmp_path, "arithmetic")
    assert report["total_artifacts"] == 2
    assert report["unique_artifacts"] == 2
    assert report["quality_issue_count"] == 0


def test_length_report_estimate_is_positive():
    assert estimated_tokens({"question": "What is 2 + 2?", "answer": "4"}, 4.0) > 0


def test_preflight_artifacts_scans_planned_rows_without_api_calls(tmp_path, monkeypatch):
    import yaml
    import slm_synth.pretrain.preflight_artifacts as preflight
    config = tmp_path / "synthetic.yaml"
    config.write_text(yaml.safe_dump({
        "output_dir": str(tmp_path / "out"),
        "target_total_tokens": 1000,
        "generation": {"batch_size": 32},
        "mix": {"arithmetic": {"architecture": "grounded", "share": 1.0, "batch_size": 32, "avg_tokens_per_sample": 100}},
    }))
    result = preflight.scan_plan(str(config))
    assert result["signals"][0]["rounded_rows"] == 32
    assert result["signals"][0]["exact_duplicates"] == 0
