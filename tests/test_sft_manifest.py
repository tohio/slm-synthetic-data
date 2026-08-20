import json

import pytest

from slm_synth.sft.manifest import build_manifest_payload, write_manifest, write_run_manifest


def _row(row_id, *, category, difficulty, template_family, eval_family):
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
            {"role": "assistant", "content": "43"},
        ],
        "metadata": {
            "task_family": eval_family,
            "interaction_modes": ["single_turn"],
            "output_mode": "free_text",
            "context_mode": "self_contained",
            "difficulty": difficulty,
            "template_family": template_family,
        },
    }


def test_build_manifest_payload_counts_sft_metadata(tmp_path):
    rows = [
        _row(
            "sft_answer_only_arithmetic_000001",
            category="answer_only_compliance",
            difficulty=1,
            template_family="direct_qa",
            eval_family="grounded_qa_and_reading",
        ),
        _row(
            "sft_direct_arithmetic_000002",
            category="direct_arithmetic",
            difficulty=2,
            template_family="direct_qa",
            eval_family="applied_math_and_reasoning",
        ),
    ]

    payload = build_manifest_payload(
        dataset_path=tmp_path / "sft.jsonl",
        rows=rows,
        generation_run="sft-smoke-001",
        metadata={"source": "unit-test"},
    )

    assert payload["dataset_type"] == "sft"
    assert payload["dataset_path"] == str(tmp_path / "sft.jsonl")
    assert payload["row_count"] == 2
    assert payload["generation_run"] == "sft-smoke-001"
    assert payload["task_families"] == {"applied_math_and_reasoning": 1, "grounded_qa_and_reading": 1}
    assert payload["template_families"] == {"direct_qa": 2}
    assert payload["difficulty_counts"] == {"1": 1, "2": 1}
    assert payload["metadata"] == {"source": "unit-test"}


def test_write_manifest_writes_sft_manifest(tmp_path):
    rows = [
        _row(
            "sft_answer_only_arithmetic_000001",
            category="answer_only_compliance",
            difficulty=1,
            template_family="direct_qa",
            eval_family="grounded_qa_and_reading",
        )
    ]

    path = write_manifest(
        manifest_path=tmp_path / "manifests" / "sft-smoke-001.manifest.json",
        dataset_path=tmp_path / "datasets" / "sft.jsonl",
        rows=rows,
        generation_run="sft-smoke-001",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_type"] == "sft"
    assert payload["row_count"] == 1
    assert payload["task_families"] == {"grounded_qa_and_reading": 1}


def test_write_run_manifest_summarizes_sft_family_outputs(tmp_path):
    manifest_path = write_run_manifest(
        manifest_path=tmp_path / "manifests" / "sft-smoke-001.manifest.json",
        generation_run="sft-smoke-001",
        datasets=[
            {
                "family": "answer_only_arithmetic",
                "dataset_path": tmp_path / "datasets" / "answer_only_arithmetic.jsonl",
                "manifest_path": tmp_path / "manifests" / "answer_only_arithmetic.sft-smoke-001.manifest.json",
                "row_count": 2,
            },
            {
                "family": "rewriting_and_editing",
                "dataset_path": tmp_path / "datasets" / "rewriting_and_editing.jsonl",
                "manifest_path": tmp_path / "manifests" / "rewriting_and_editing.sft-smoke-001.manifest.json",
                "row_count": 3,
            },
        ],
        metadata={"family_count": 2},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "sft"
    assert manifest["generation_run"] == "sft-smoke-001"
    assert manifest["families"] == ["answer_only_arithmetic", "rewriting_and_editing"]
    assert manifest["total_rows"] == 5
    assert manifest["metadata"] == {"family_count": 2}
    assert manifest["datasets"][0]["dataset_path"] == str(tmp_path / "datasets" / "answer_only_arithmetic.jsonl")
    assert manifest["datasets"][0]["manifest_path"] == str(
        tmp_path / "manifests" / "answer_only_arithmetic.sft-smoke-001.manifest.json"
    )


def test_write_run_manifest_rejects_duplicate_sft_families(tmp_path):
    with pytest.raises(ValueError, match="duplicate family"):
        write_run_manifest(
            manifest_path=tmp_path / "manifest.json",
            generation_run="sft-smoke-001",
            datasets=[
                {
                    "family": "answer_only_arithmetic",
                    "dataset_path": tmp_path / "a.jsonl",
                    "manifest_path": tmp_path / "a.manifest.json",
                    "row_count": 1,
                },
                {
                    "family": "answer_only_arithmetic",
                    "dataset_path": tmp_path / "b.jsonl",
                    "manifest_path": tmp_path / "b.manifest.json",
                    "row_count": 1,
                },
            ],
        )


def test_build_manifest_payload_rejects_invalid_rows(tmp_path):
    row = _row(
        "sft_answer_only_arithmetic_000001",
        category="answer_only_compliance",
        difficulty=1,
        template_family="direct_qa",
        eval_family="grounded_qa_and_reading",
    )
    row["metadata"]["failure_mode"] = "extra_explanation"

    with pytest.raises(ValueError, match="unsupported field"):
        build_manifest_payload(
            dataset_path=tmp_path / "sft.jsonl",
            rows=[row],
            generation_run="sft-smoke-001",
        )
