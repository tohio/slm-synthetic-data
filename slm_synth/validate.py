from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_synth.paths import raw_dir_from_config
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


def validate_file(path: Path, out_dir: Path, reject_dir: Path) -> tuple[int, int]:
    accepted = 0
    rejected = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = None
            try:
                obj = json.loads(line)
                record_type = obj["type"]
                validator = VALIDATORS[record_type]
                validator(obj)
            except Exception as exc:
                fallback_type = "unknown"
                if isinstance(obj, dict):
                    fallback_type = str(obj.get("type", "unknown"))

                with open(reject_dir / f"{fallback_type}.jsonl", "a", encoding="utf-8") as rej:
                    wrapped = {
                        "source_file": path.name,
                        "line": line_no,
                        "error": str(exc),
                        "raw": line,
                    }
                    rej.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            with open(out_dir / f"{record_type}.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            accepted += 1

    return accepted, rejected


def main(raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    validated_dir = raw_dir.parent / "validated"
    rejected_dir = raw_dir.parent / "rejected"

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    validated_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild validation outputs so reruns do not append duplicates.
    for old in validated_dir.glob("*.jsonl"):
        old.unlink()

    # Remove validation-created reject files. Generation reject files are also
    # signal-named, but empty successful generation files are safe to reset.
    for old in rejected_dir.glob("*.jsonl"):
        old.unlink()

    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in raw_dir: {raw_dir}")

    total_accepted = 0
    total_rejected = 0
    for file in files:
        accepted, rejected = validate_file(file, validated_dir, rejected_dir)
        total_accepted += accepted
        total_rejected += rejected
        print(f"[validate] {file.name}: accepted={accepted}, rejected={rejected}")

    print(f"[validate] Completed accepted={total_accepted}, rejected={total_rejected}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic JSONL records.")
    parser.add_argument("raw_dir", nargs="?", help="Path to raw JSONL directory.")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml.")
    args = parser.parse_args()

    if args.config:
        raw_dir = raw_dir_from_config(args.config)
    elif args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        parser.error("provide either raw_dir or --config")

    main(raw_dir)


if __name__ == "__main__":
    cli()
