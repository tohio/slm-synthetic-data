from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE, iter_jsonl, signal_to_filename, validate_record


def _signals_from_args(raw_dir: Path, signal: str | None) -> list[str]:
    if signal:
        return [signal]
    signals: list[str] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        signals.append(SIGNAL_FROM_FILE.get(path.name, path.stem))
    return signals


def validate_signal(raw_dir: Path, validated_dir: Path, rejected_dir: Path, signal: str) -> tuple[int, int]:
    src = raw_dir / signal_to_filename(signal)
    if not src.exists():
        print(f"[validate] {src.name}: missing, skipped")
        return 0, 0

    validated_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    dst = validated_dir / src.name
    rej = rejected_dir / src.name

    accepted = 0
    rejected = 0
    with dst.open("w", encoding="utf-8") as out, rej.open("w", encoding="utf-8") as bad:
        for lineno, row, parse_issues in iter_jsonl(src):
            if parse_issues:
                bad.write(json.dumps({"line": lineno, "issues": parse_issues, "row": row}, ensure_ascii=False) + "\n")
                rejected += 1
                continue
            assert row is not None
            result = validate_record(
                signal,
                row,
                require_arithmetic_verification=(signal == "arithmetic"),
                require_mcq_verification=(signal == "educational_qa_mcq_math"),
            )
            if result.ok and result.record is not None:
                out.write(json.dumps(result.record, ensure_ascii=False) + "\n")
                accepted += 1
            else:
                bad.write(json.dumps({"line": lineno, "issues": result.issues, "row": row}, ensure_ascii=False) + "\n")
                rejected += 1

    print(f"[validate] {src.name}: accepted={accepted}, rejected={rejected}")
    return accepted, rejected


def run_from_config(config_path: str, signal: str | None = None) -> None:
    cfg = load_yaml_config(config_path)
    out = resolve_output_dir(cfg)
    raw_dir = out / "raw"
    validated_dir = out / "validated"
    rejected_dir = out / "rejected"

    total_accepted = 0
    total_rejected = 0
    for sig in _signals_from_args(raw_dir, signal):
        accepted, rejected = validate_signal(raw_dir, validated_dir, rejected_dir, sig)
        total_accepted += accepted
        total_rejected += rejected

    print(f"[validate] Completed accepted={total_accepted}, rejected={total_rejected}")


def run_positional(raw_dir: str, validated_dir: str, rejected_dir: str, signal: str | None = None) -> None:
    raw = Path(raw_dir)
    validated = Path(validated_dir)
    rejected = Path(rejected_dir)
    total_accepted = 0
    total_rejected = 0
    for sig in _signals_from_args(raw, signal):
        accepted, rejected_count = validate_signal(raw, validated, rejected, sig)
        total_accepted += accepted
        total_rejected += rejected_count
    print(f"[validate] Completed accepted={total_accepted}, rejected={total_rejected}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic JSONL records by signal schema")
    parser.add_argument("raw_dir", nargs="?", help="Legacy raw input directory")
    parser.add_argument("validated_dir", nargs="?", help="Legacy validated output directory")
    parser.add_argument("rejected_dir", nargs="?", help="Legacy rejected output directory")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml")
    parser.add_argument("--signal", default=None, help="Optional signal filter")
    args = parser.parse_args()

    if args.config:
        run_from_config(args.config, signal=args.signal)
        return

    if not (args.raw_dir and args.validated_dir and args.rejected_dir):
        parser.error("Either --config or raw_dir validated_dir rejected_dir is required")
    run_positional(args.raw_dir, args.validated_dir, args.rejected_dir, signal=args.signal)


if __name__ == "__main__":
    cli()
