import json

import pytest

from slm_synth.dpo.manifest import build_manifest_payload, write_manifest


def _row(row_id, *, category, failure_mode, difficulty, template_family, eval_family):
    return {
        "id": row_id,
        "prompt": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
        ],
        "chosen": [
            {"role": "assistant", "content": "43"},
        ],
        "rejected": [
            {"role": "assistant", "content": "The answer is 43 because 16 plus 27 equals 43."},
        ],
        "metadata": {
            "category": category,
            "failure_mode": failure_mode,
            "difficulty": difficulty,
            "template_family": template_family,
            "eval_family": eval_family,
        },
    }


def test_build_manifest_payload_counts_dpo_metadata(tmp_path):
    rows = [
        _row(
            "dpo_answer_only_arithmetic_000001",
            category="answer_only_compliance",
            failure_mode="extra_explanation",
            difficulty=1,
            template_family="direct_qa",
            eval_family="basic_arithmetic_qa",
        ),
        _row(
            "dpo_format_repeat_000002",
            category="exact_output_format_control",
            failure_mode="format_violation",
            difficulty=2,
            template_family="repeat_word_count",
            eval_family="repeat_exact_n_times",
        ),
    ]

    payload = build_manifest_payload(
        dataset_path=tmp_path / "dpo.jsonl",
        rows=rows,
        generation_run="dpo-smoke-001",
        metadata={"source": "unit-test"},
    )

    assert payload["dataset_type"] == "dpo"
    assert payload["dataset_path"] == str(tmp_path / "dpo.jsonl")
    assert payload["row_count"] == 2
    assert payload["generation_run"] == "dpo-smoke-001"
    assert payload["categories"] == {"answer_only_compliance": 1, "exact_output_format_control": 1}
    assert payload["eval_families"] == {"basic_arithmetic_qa": 1, "repeat_exact_n_times": 1}
    assert payload["template_families"] == {"direct_qa": 1, "repeat_word_count": 1}
    assert payload["difficulty_counts"] == {"1": 1, "2": 1}
    assert payload["failure_modes"] == {"extra_explanation": 1, "format_violation": 1}
    assert payload["metadata"] == {"source": "unit-test"}


def test_write_manifest_writes_dpo_manifest(tmp_path):
    rows = [
        _row(
            "dpo_answer_only_arithmetic_000001",
            category="answer_only_compliance",
            failure_mode="extra_explanation",
            difficulty=1,
            template_family="direct_qa",
            eval_family="basic_arithmetic_qa",
        )
    ]

    path = write_manifest(
        manifest_path=tmp_path / "manifests" / "dpo-smoke-001.manifest.json",
        dataset_path=tmp_path / "datasets" / "dpo.jsonl",
        rows=rows,
        generation_run="dpo-smoke-001",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_type"] == "dpo"
    assert payload["row_count"] == 1
    assert payload["failure_modes"] == {"extra_explanation": 1}


def test_build_manifest_payload_rejects_invalid_rows(tmp_path):
    row = _row(
        "dpo_answer_only_arithmetic_000001",
        category="answer_only_compliance",
        failure_mode="extra_explanation",
        difficulty=1,
        template_family="direct_qa",
        eval_family="basic_arithmetic_qa",
    )
    row["rejected"] = list(row["chosen"])

    with pytest.raises(ValueError, match="must differ"):
        build_manifest_payload(
            dataset_path=tmp_path / "dpo.jsonl",
            rows=[row],
            generation_run="dpo-smoke-001",
        )
