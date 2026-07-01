import json

import pytest

from slm_synth.sft.io import read_jsonl, write_jsonl


def _row():
    return {
        "id": "sft_answer_only_arithmetic_000001",
        "messages": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
            {"role": "assistant", "content": "43"},
        ],
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
        },
    }


def test_write_jsonl_writes_validated_sft_rows(tmp_path):
    path = tmp_path / "nested" / "sft.jsonl"

    count = write_jsonl([_row()], path)

    assert count == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [_row()]


def test_write_jsonl_rejects_invalid_sft_rows(tmp_path):
    row = _row()
    row["metadata"]["failure_mode"] = "extra_explanation"

    with pytest.raises(ValueError, match="unsupported field"):
        write_jsonl([row], tmp_path / "sft.jsonl")


def test_read_jsonl_reads_and_validates_rows(tmp_path):
    path = tmp_path / "sft.jsonl"
    write_jsonl([_row()], path)

    assert read_jsonl(path) == [_row()]


def test_read_jsonl_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSONL"):
        read_jsonl(path)


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text("\n" + json.dumps(_row()) + "\n\n", encoding="utf-8")

    assert read_jsonl(path) == [_row()]
