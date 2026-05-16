import json
from pathlib import Path
from slm_synth.validate import validate_file

def test_validate_file(tmp_path):
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "out"
    rej = tmp_path / "rej"
    out.mkdir()
    rej.mkdir()

    valid = {"type":"arithmetic","question":"Q","steps":["S"],"answer":"A"}
    invalid = {"type":"arithmetic","question":123}

    with open(raw, "w") as f:
        f.write(json.dumps(valid) + "\n")
        f.write(json.dumps(invalid) + "\n")

    validate_file(raw, out, rej)

    assert (out / "arithmetic.jsonl").exists()
    assert (rej / "arithmetic.jsonl").exists()
