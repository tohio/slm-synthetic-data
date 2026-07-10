import json

import pytest

from slm_synth.sft.io import write_jsonl
from slm_synth.sft.report import build_coverage_report, write_coverage_report


def _sft_row(*, row_id: str, category: str, eval_family: str, template_family: str) -> dict[str, object]:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "Answer with only the final value."},
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
