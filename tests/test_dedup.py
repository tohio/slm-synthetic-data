import json

from slm_synth.dedup import dedup_signal


def test_exact_dedup_general_mcq(tmp_path):
    validated = tmp_path / "validated"
    deduped = tmp_path / "deduped"
    validated.mkdir()
    record = {
        "type": "educational_qa_mcq_general",
        "question": "Which word is an adverb in 'Mina quickly packed the box'?",
        "choices": ["Mina", "quickly", "packed", "box"],
        "correct_index": 1,
        "explanation": "Quickly describes the action.",
    }
    path = validated / "educational_qa_mcq_general.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps(record) + "\n")

    kept, dropped = dedup_signal(validated, deduped, "educational_qa_mcq_general")
    assert kept == 1
    assert dropped == 1
