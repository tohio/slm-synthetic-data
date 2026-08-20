import json

import pytest

from slm_synth.dpo.manifest import build_manifest_payload, write_manifest, write_run_manifest


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
            "task_family": "applied_math_and_reasoning",
            "interaction_modes": ["single_turn"],
            "output_mode": "concise",
            "context_mode": "self_contained",
            "preference_dimension": eval_family,
            "failure_mode": failure_mode,
            "difficulty": difficulty,
            "template_family": template_family,
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
            eval_family="factual_accuracy",
        ),
        _row(
            "dpo_format_repeat_000002",
            category="exact_output_format_control",
            failure_mode="format_violation",
            difficulty=2,
            template_family="repeat_word_count",
            eval_family="instruction_adherence",
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
    assert payload["task_families"] == {"applied_math_and_reasoning": 2}
    assert payload["preference_dimensions"] == {"factual_accuracy": 1, "instruction_adherence": 1}
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
            eval_family="factual_accuracy",
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


def test_write_run_manifest_summarizes_dpo_dimension_outputs(tmp_path):
    manifest_path = write_run_manifest(
        manifest_path=tmp_path / "manifests" / "dpo-smoke-001.manifest.json",
        generation_run="dpo-smoke-001",
        datasets=[
            {
                "preference_dimension": "answer_only_arithmetic",
                "dataset_path": tmp_path / "datasets" / "answer_only_arithmetic.jsonl",
                "manifest_path": tmp_path / "manifests" / "answer_only_arithmetic.dpo-smoke-001.manifest.json",
                "row_count": 2,
            },
            {
                "preference_dimension": "instruction_adherence",
                "dataset_path": tmp_path / "datasets" / "instruction_adherence.jsonl",
                "manifest_path": tmp_path / "manifests" / "instruction_adherence.dpo-smoke-001.manifest.json",
                "row_count": 3,
            },
        ],
        metadata={"preference_dimension_count": 2},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "dpo"
    assert manifest["generation_run"] == "dpo-smoke-001"
    assert manifest["preference_dimensions"] == ["answer_only_arithmetic", "instruction_adherence"]
    assert manifest["total_rows"] == 5
    assert manifest["metadata"] == {"preference_dimension_count": 2}
    assert manifest["datasets"][0]["dataset_path"] == str(tmp_path / "datasets" / "answer_only_arithmetic.jsonl")
    assert manifest["datasets"][0]["manifest_path"] == str(
        tmp_path / "manifests" / "answer_only_arithmetic.dpo-smoke-001.manifest.json"
    )


def test_write_run_manifest_rejects_duplicate_dpo_dimensions(tmp_path):
    with pytest.raises(ValueError, match="duplicate preference dimension"):
        write_run_manifest(
            manifest_path=tmp_path / "manifest.json",
            generation_run="dpo-smoke-001",
            datasets=[
                {
                    "preference_dimension": "answer_only_arithmetic",
                    "dataset_path": tmp_path / "a.jsonl",
                    "manifest_path": tmp_path / "a.manifest.json",
                    "row_count": 1,
                },
                {
                    "preference_dimension": "answer_only_arithmetic",
                    "dataset_path": tmp_path / "b.jsonl",
                    "manifest_path": tmp_path / "b.manifest.json",
                    "row_count": 1,
                },
            ],
        )


def test_build_manifest_payload_rejects_invalid_rows(tmp_path):
    row = _row(
        "dpo_answer_only_arithmetic_000001",
        category="answer_only_compliance",
        failure_mode="extra_explanation",
        difficulty=1,
        template_family="direct_qa",
        eval_family="factual_accuracy",
    )
    row["rejected"] = list(row["chosen"])

    with pytest.raises(ValueError, match="must differ"):
        build_manifest_payload(
            dataset_path=tmp_path / "dpo.jsonl",
            rows=[row],
            generation_run="dpo-smoke-001",
        )
