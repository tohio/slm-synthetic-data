import json

from slm_synth.distillation_sft.response_diversity import (
    build_response_diversity_summary,
    normalize_response_text,
)


def _row(row_id: str, *, prompt: str, response: str) -> dict[str, object]:
    return {
        "id": row_id,
        "prompt": prompt,
        "reasoning": None,
        "response": response,
        "metadata": {
            "category": "general_instruction_following",
            "difficulty": 2,
            "template_family": "python_optional_key_bug",
            "eval_family": None,
        },
    }


def _write_rows(path, responses):
    rows = [
        _row(f"{path.stem}-{index}", prompt=f"Prompt {index}", response=response)
        for index, response in enumerate(responses)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_normalize_response_text_ignores_case_and_whitespace():
    assert normalize_response_text("  A\n  concise RESPONSE. ") == "a concise response."


def test_build_response_diversity_summary_reports_each_signal(tmp_path):
    debugging = tmp_path / "debugging.jsonl"
    planning = tmp_path / "planning.jsonl"
    _write_rows(debugging, ["same", "same", "different", "another"])
    _write_rows(planning, ["plan one", "plan two"])

    summary = build_response_diversity_summary([debugging, planning])

    assert summary["row_count"] == 6
    assert summary["unique_response_count"] == 5
    assert summary["duplicate_response_count"] == 1
    assert summary["signals"]["debugging"] == {
        "row_count": 4,
        "unique_response_count": 3,
        "duplicate_response_count": 1,
        "unique_response_ratio": 0.75,
        "duplicate_examples": [{"response": "same", "count": 2}],
    }
    assert summary["signals"]["planning"]["unique_response_ratio"] == 1.0
