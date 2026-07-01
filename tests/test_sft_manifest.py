import json

import pytest

from slm_synth.sft.manifest import build_manifest_payload, write_manifest


def _row(row_id, *, category, difficulty, template_family, eval_family):
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
            {"role": "assistant", "content": "43"},
        ],
        "metadata": {
            "category": category,
            "difficulty": difficulty,
            "template_family": template_family,
            "eval_family": eval_family,
        },
    }


def test_build_manifest_payload_counts_sft_metadata(tmp_path):
    rows = [
        _row(
            "sft_answer_only_arithmetic_000001",
            category="answer_only_compliance",
            difficulty=1,
            template_family="direct_qa",
            eval_family="basic_arithmetic_qa",
        ),
        _row(
            "sft_direct_arithmetic_000002",
            category="direct_arithmetic",
            difficulty=2,
            template_family="direct_qa",
            eval_family="direct_subtraction",
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
    assert payload["categories"] == {"answer_only_compliance": 1, "direct_arithmetic": 1}
    assert payload["eval_families"] == {"basic_arithmetic_qa": 1, "direct_subtraction": 1}
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
            eval_family="basic_arithmetic_qa",
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
    assert payload["categories"] == {"answer_only_compliance": 1}


def test_build_manifest_payload_rejects_invalid_rows(tmp_path):
    row = _row(
        "sft_answer_only_arithmetic_000001",
        category="answer_only_compliance",
        difficulty=1,
        template_family="direct_qa",
        eval_family="basic_arithmetic_qa",
    )
    row["metadata"]["failure_mode"] = "extra_explanation"

    with pytest.raises(ValueError, match="unsupported field"):
        build_manifest_payload(
            dataset_path=tmp_path / "sft.jsonl",
            rows=[row],
            generation_run="sft-smoke-001",
        )
