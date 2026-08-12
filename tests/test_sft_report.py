import json

import pytest

from slm_synth.sft.io import write_jsonl
from slm_synth.sft.report import build_coverage_report, require_publish_ready_report, write_coverage_report
from slm_synth.taxonomy.holdouts import HoldoutRecord, HoldoutRegistry


def _sft_row(
    *,
    row_id: str,
    category: str,
    eval_family: str,
    template_family: str,
    prompt: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": prompt or f"Answer item {row_id} with only the final value."},
            {"role": "assistant", "content": "4"},
        ],
        "metadata": {
            "category": category,
            "difficulty": 1,
            "template_family": template_family,
            "eval_family": eval_family,
        },
    }


def test_build_sft_coverage_report_counts_metadata_across_files(tmp_path):
    arithmetic_path = tmp_path / "arithmetic.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="sft-arithmetic-1",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
            ),
            _sft_row(
                row_id="sft-arithmetic-2",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
            ),
        ],
        arithmetic_path,
    )
    write_jsonl(
        [
            _sft_row(
                row_id="sft-repeat-1",
                category="exact_output_format_control",
                eval_family="repeat_exact_n_times",
                template_family="repeat_word_count",
            )
        ],
        repeat_path,
    )

    report = build_coverage_report([tmp_path])

    assert report["dataset_type"] == "sft"
    assert report["row_count"] == 3
    assert report["files"] == {
        str(arithmetic_path): 2,
        str(repeat_path): 1,
    }
    assert report["categories"] == {
        "answer_only_compliance": 2,
        "exact_output_format_control": 1,
    }
    assert report["eval_families"] == {
        "basic_arithmetic_qa": 2,
        "repeat_exact_n_times": 1,
    }
    assert report["template_families"] == {
        "direct_qa": 2,
        "repeat_word_count": 1,
    }
    assert report["difficulty_counts"] == {"1": 3}
    assert report["families"]["basic_arithmetic_qa"]["acceptance"] == {
        "attempted_rows": 2,
        "accepted_rows": 2,
        "rejected_rows": 0,
        "duplicate_rows": 0,
        "remaining_rows": 0,
        "publish_ready": False,
        "publish_blockers": ["holdouts_not_checked"],
    }


def test_write_sft_coverage_report_writes_json(tmp_path):
    report_path = tmp_path / "reports" / "sft_coverage.json"
    report = {
        "dataset_type": "sft",
        "row_count": 0,
        "files": {},
        "categories": {},
        "eval_families": {},
        "template_families": {},
        "difficulty_counts": {},
    }

    written = write_coverage_report(report=report, path=report_path)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_build_sft_coverage_report_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="input path does not exist"):
        build_coverage_report([tmp_path / "missing.jsonl"])


def test_build_sft_coverage_report_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no JSONL dataset files found"):
        build_coverage_report([tmp_path])


def test_sft_report_blocks_normalized_prompt_and_conversation_duplicates(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
                prompt="What is 2 + 2?",
            ),
            _sft_row(
                row_id="second",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
                prompt="  what IS 2 + 2? ",
            ),
        ],
        dataset_path,
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
    )

    assert report["content_uniqueness"]["prompts"]["duplicate_count"] == 1
    assert report["content_uniqueness"]["conversations"]["duplicate_count"] == 1
    assert report["acceptance"]["duplicate_rows"] == 1
    assert report["acceptance"]["remaining_rows"] == 1
    assert report["acceptance"]["publish_ready"] is False
    assert "duplicate_prompts" in report["acceptance"]["publish_blockers"]


def test_sft_report_checks_holdouts_and_keeps_response_repetition_reporting_only(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
                prompt="What is 2 + 2?",
            ),
            _sft_row(
                row_id="second",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
                prompt="What is 3 + 1?",
            ),
        ],
        dataset_path,
    )
    clean_registry = HoldoutRegistry([])

    report = build_coverage_report([dataset_path], holdout_registry=clean_registry)

    assert report["holdouts"] == {
        "status": "checked",
        "collision_count": 0,
        "collision_ids": [],
    }
    assert report["content_uniqueness"]["responses"]["duplicate_count"] == 1
    assert report["acceptance"]["publish_ready"] is True
    require_publish_ready_report(report)

    collision_registry = HoldoutRegistry(
        [
            HoldoutRecord(
                id="holdout-1",
                eval_family="basic_arithmetic_qa",
                prompt="what is 2 + 2?",
                answer="4",
            )
        ]
    )
    collision_report = build_coverage_report(
        [dataset_path],
        holdout_registry=collision_registry,
    )
    assert collision_report["holdouts"]["collision_ids"] == ["first"]
    assert "eval_holdout_collisions" in collision_report["acceptance"]["publish_blockers"]


def test_sft_report_marks_missing_holdout_check_as_publish_blocker(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
            )
        ],
        dataset_path,
    )

    report = build_coverage_report([dataset_path])

    assert report["holdouts"]["status"] == "not_checked"
    assert report["acceptance"]["publish_blockers"] == ["holdouts_not_checked"]
    with pytest.raises(ValueError, match="holdouts_not_checked"):
        require_publish_ready_report(report)


def test_sft_report_detects_files_underfilled_against_complete_manifest(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="basic_arithmetic_qa",
                template_family="direct_qa",
            )
        ],
        dataset_path,
    )
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "publish_ready": True,
                    "planned_rows": 2,
                    "rejection_reason_counts": {"batch_acceptance_error": 1},
                    "accepted_target": {
                        "target": 2,
                        "accepted": 2,
                        "attempted": 2,
                        "remaining": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=manifest_path,
    )

    assert report["acceptance"]["accepted_rows"] == 1
    assert report["acceptance"]["rejection_reason_counts"] == {"batch_acceptance_error": 1}
    assert report["acceptance"]["remaining_rows"] == 1
    assert "accepted_target_underfilled" in report["acceptance"]["publish_blockers"]
