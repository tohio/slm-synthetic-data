import json

from slm_synth.pretrain.validate import validate_signal


def test_validate_signal_writes_math_mcq_without_raw_verification_fields(tmp_path):
    raw_dir = tmp_path / "raw"
    validated_dir = tmp_path / "validated"
    rejected_dir = tmp_path / "rejected"
    raw_dir.mkdir()

    record = {
        "type": "educational_qa_mcq_math",
        "question": "What is 25% of 120?",
        "choices": ["15", "20", "30", "25"],
        "correct_index": 3,
        "explanation": "25% of 120 is 30 because (120 * 25) / 100 = 30.",
        "verification_expression": "(120 * 25) / 100",
        "verification_answer": "30",
    }
    with (raw_dir / "educational_qa_mcq_math.jsonl").open("w") as handle:
        handle.write(json.dumps(record) + "\n")

    accepted, rejected = validate_signal(raw_dir, validated_dir, rejected_dir, "educational_qa_mcq_math")
    assert accepted == 1
    assert rejected == 0
    output = json.loads((validated_dir / "educational_qa_mcq_math.jsonl").read_text().strip())
    assert output["correct_index"] == 2
    assert "verification_expression" not in output
