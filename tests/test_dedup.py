import json
from pathlib import Path
from slm_synth.dedup import dedup_file

def test_dedup(tmp_path):
    file = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"

    obj = {"type":"arithmetic","question":"Q","steps":["S"],"answer":"A"}

    with open(file, "w") as f:
        f.write(json.dumps(obj) + "\n")
        f.write(json.dumps(obj) + "\n")  # duplicate

    dedup_file(file, out, threshold=0.85)

    with open(out, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
