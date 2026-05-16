import json
from pathlib import Path
from slm_synth.schemas import (
    validate_arithmetic,
    validate_task_code,
    validate_educational_qa_mcq,
    validate_factual_restraint,
)


VALIDATORS = {
    "arithmetic": validate_arithmetic,
    "task_code": validate_task_code,
    "educational_qa_mcq": validate_educational_qa_mcq,
    "factual_restraint": validate_factual_restraint,
}


def validate_file(path: Path, out_dir: Path, reject_dir: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj["type"]

            try:
                VALIDATORS[t](obj)
                with open(out_dir / f"{t}.jsonl", "a", encoding="utf-8") as out:
                    out.write(json.dumps(obj) + "\n")
            except Exception:
                with open(reject_dir / f"{t}.jsonl", "a", encoding="utf-8") as rej:
                    rej.write(json.dumps(obj) + "\n")


def main(raw_dir: str):
    raw_dir = Path(raw_dir)
    validated = raw_dir.parent / "validated"
    rejected = raw_dir.parent / "rejected"

    validated.mkdir(exist_ok=True)
    rejected.mkdir(exist_ok=True)

    for file in raw_dir.glob("*.jsonl"):
        validate_file(file, validated, rejected)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
