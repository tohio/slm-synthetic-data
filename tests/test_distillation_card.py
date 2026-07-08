import json

import pytest

from slm_synth.distillation.card import render_dataset_card, write_dataset_card


def _run_manifest():
    return {
        "schema_version": 1,
        "generation_run": "smoke-001",
        "teacher_model": "openai/gpt-4.1-mini",
        "teacher_provider": "openrouter",
        "token_target": "100K",
        "signals": ["cloud", "database"],
        "datasets": [
            {
                "signal": "cloud",
                "dataset_path": "data/distillation/datasets/cloud.jsonl",
                "manifest_path": "data/distillation/manifests/cloud.smoke-001.manifest.json",
                "row_count": 2,
            },
            {
                "signal": "database",
                "dataset_path": "data/distillation/datasets/database.jsonl",
                "manifest_path": "data/distillation/manifests/database.smoke-001.manifest.json",
                "row_count": 3,
            },
        ],
        "total_rows": 5,
        "metadata": {
            "signal_count": 2,
            "target_rows": 5,
            "planned_prompt_rows": 5,
            "accepted_rows": 5,
            "rejected_rows": 0,
        },
    }


def test_render_dataset_card_includes_run_provenance_and_schema():
    text = render_dataset_card(
        run_manifest=_run_manifest(),
        dataset_name="SLM Synthetic Distillation Smoke",
        license_name="mit",
    )

    assert "# SLM Synthetic Distillation Smoke" in text
    assert 'license: "mit"' in text
    assert "- Generation run: `smoke-001`" in text
    assert "- Teacher provider: `openrouter`" in text
    assert "- Teacher model: `openai/gpt-4.1-mini`" in text
    assert "- Target rows: `5`" in text
    assert "- Planned prompt rows: `5`" in text
    assert "- Accepted rows: `5`" in text
    assert "- Rejected rows: `0`" in text
    assert "| cloud | 2 | `data/distillation/datasets/cloud.jsonl` |" in text
    assert '"id": "string"' in text
    assert '"reasoning": null' in text
    assert "`reasoning` may also be a list of strings" not in text
    assert "intentionally excluded from public training rows" in text


def test_write_dataset_card_reads_manifest_and_writes_markdown(tmp_path):
    manifest_path = tmp_path / "smoke-001.manifest.json"
    output_path = tmp_path / "README.md"
    manifest_path.write_text(json.dumps(_run_manifest()), encoding="utf-8")

    result = write_dataset_card(
        run_manifest_path=manifest_path,
        output_path=output_path,
        dataset_name="SLM Synthetic Distillation Smoke",
    )

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "# SLM Synthetic Distillation Smoke" in text
    assert "- Total rows: `5`" in text


def test_render_dataset_card_rejects_non_openrouter_manifest_provider():
    manifest = _run_manifest()
    manifest["teacher_provider"] = "groq"

    with pytest.raises(ValueError, match="openrouter"):
        render_dataset_card(run_manifest=manifest, dataset_name="Dataset")
