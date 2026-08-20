import json

import pytest

from slm_synth.dpo.io import write_jsonl
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.dpo.report import require_publish_ready_report
from slm_synth.taxonomy.holdouts import HoldoutRecord, HoldoutRegistry


def _dpo_row(
    *,
    row_id: str,
    category: str,
    eval_family: str,
    template_family: str,
    failure_mode: str,
    prompt: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": prompt or f"Answer item {row_id} with only the final value."}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "The answer is 4."}],
        "metadata": {
            "task_family": "applied_math_and_reasoning",
            "interaction_modes": ["single_turn"],
            "output_mode": "concise",
            "context_mode": "self_contained",
            "preference_dimension": eval_family,
            "difficulty": 1,
            "template_family": template_family,
            "failure_mode": failure_mode,
        },
    }


def test_build_dpo_coverage_report_counts_metadata_across_files(tmp_path):
    arithmetic_path = tmp_path / "arithmetic.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    write_jsonl(
        [
            _dpo_row(
                row_id="dpo-arithmetic-1",
                category="answer_only_compliance",
                eval_family="factual_accuracy",
                template_family="direct_qa",
                failure_mode="extra_explanation",
            ),
            _dpo_row(
                row_id="dpo-arithmetic-2",
                category="answer_only_compliance",
                eval_family="factual_accuracy",
                template_family="direct_qa",
                failure_mode="extra_explanation",
            ),
        ],
        arithmetic_path,
    )
    write_jsonl(
        [
            _dpo_row(
                row_id="dpo-repeat-1",
                category="exact_output_format_control",
                eval_family="instruction_adherence",
                template_family="repeat_word_count",
                failure_mode="format_violation",
            )
        ],
        repeat_path,
    )

    report = build_coverage_report([tmp_path])

    assert report["dataset_type"] == "dpo"
    assert report["row_count"] == 3
    assert report["files"] == {
        str(arithmetic_path): 2,
        str(repeat_path): 1,
    }
    assert report["preference_dimensions"] == {
        "factual_accuracy": 2,
        "instruction_adherence": 1,
    }
    assert report["template_families"] == {
        "direct_qa": 2,
        "repeat_word_count": 1,
    }
    assert report["difficulty_counts"] == {"1": 3}
    assert report["failure_modes"] == {
        "extra_explanation": 2,
        "format_violation": 1,
    }
    assert report["acceptance"]["publish_blockers"] == ["holdouts_not_checked"]
    assert report["families"]["factual_accuracy"]["content_quality"]["prompts"]["unique"] == 2


def test_write_dpo_coverage_report_writes_json(tmp_path):
    report_path = tmp_path / "reports" / "dpo_coverage.json"
    report = {
        "dataset_type": "dpo",
        "row_count": 0,
        "files": {},
        "task_families": {},
        "preference_dimensions": {},
        "template_families": {},
        "difficulty_counts": {},
        "failure_modes": {},
    }

    written = write_coverage_report(report=report, path=report_path)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_build_dpo_coverage_report_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="input path does not exist"):
        build_coverage_report([tmp_path / "missing.jsonl"])


def test_build_dpo_coverage_report_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no JSONL dataset files found"):
        build_coverage_report([tmp_path])


def test_dpo_report_blocks_normalized_duplicates_and_holdout_collisions(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _dpo_row(
                row_id="one", category="answer_only_compliance", eval_family="factual_accuracy",
                template_family="direct_qa", failure_mode="wrong_numeric_answer", prompt="What is 2 + 2?",
            ),
            _dpo_row(
                row_id="two", category="answer_only_compliance", eval_family="factual_accuracy",
                template_family="direct_qa", failure_mode="wrong_numeric_answer", prompt=" what IS 2 + 2? ",
            ),
        ],
        dataset_path,
    )
    registry = HoldoutRegistry([
        HoldoutRecord(
            id="held-out", eval_family="factual_accuracy",
            prompt="What is 2 + 2?", answer="4",
        )
    ])

    report = build_coverage_report([dataset_path], holdout_registry=registry)

    assert report["content_quality"]["prompts"]["duplicate_count"] == 1
    assert report["content_quality"]["triples"]["duplicate_count"] == 1
    assert report["acceptance"]["duplicate_pairs"] == 1
    assert report["acceptance"]["remaining_pairs"] == 1
    assert report["holdouts"]["collision_ids"] == ["one", "two"]
    assert report["acceptance"]["publish_blockers"] == [
        "duplicate_prompts", "duplicate_triples", "eval_holdout_collisions", "accepted_target_underfilled",
    ]


def test_dpo_report_uses_manifest_accounting_and_accepts_clean_dataset(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl([
        _dpo_row(
            row_id="one", category="answer_only_compliance", eval_family="factual_accuracy",
            template_family="direct_qa", failure_mode="wrong_numeric_answer",
        )
    ], dataset_path)
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {
        "publish_ready": True,
        "planned_pairs": 1,
        "attempted_pairs": 2,
        "accepted_target": {"target": 1, "accepted": 1, "attempted": 2, "remaining": 0},
        "duplicate_pairs": 1,
        "duplicate_reason_counts": {"duplicate_prompt": 1},
        "pairs_per_family": {"factual_accuracy": 1},
    }}), encoding="utf-8")

    report = build_coverage_report(
        [dataset_path], holdout_registry=HoldoutRegistry([]), run_manifest=manifest_path,
    )

    assert report["acceptance"]["attempted_pairs"] == 2
    assert report["acceptance"]["accepted_pairs"] == 1
    assert report["acceptance"]["duplicate_pairs"] == 1
    assert report["acceptance"]["remaining_pairs"] == 0
    assert report["acceptance"]["publish_ready"] is True
    require_publish_ready_report(report)
