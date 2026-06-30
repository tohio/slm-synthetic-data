from __future__ import annotations

import argparse
from pathlib import Path

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE, audit_jsonl_file, signal_to_filename


def audit_stage(config_path: str, stage: str, signal: str | None = None, fail_on_invalid: bool = False) -> int:
    cfg = load_yaml_config(config_path)
    out = resolve_output_dir(cfg)
    stage_dir = out / stage
    signals = [signal] if signal else [SIGNAL_FROM_FILE.get(p.name, p.stem) for p in sorted(stage_dir.glob("*.jsonl"))]

    total_invalid = 0
    for sig in signals:
        path = stage_dir / signal_to_filename(sig)
        if not path.exists():
            print(f"[audit] {stage}/{path.name}: missing")
            total_invalid += 1
            continue
        result = audit_jsonl_file(path, sig)
        total_invalid += result.invalid
        issue_summary = ", ".join(f"{k}={v}" for k, v in sorted(result.issues.items())) or "none"
        print(
            f"[audit] {stage}/{path.name}: total={result.total}, valid={result.valid}, "
            f"invalid={result.invalid}, bad_json={result.bad_json}, issues={issue_summary}"
        )

    if fail_on_invalid and total_invalid:
        raise SystemExit(f"[audit] failed: invalid_records={total_invalid}")
    return total_invalid


def cli() -> None:
    parser = argparse.ArgumentParser(description="Audit synthetic JSONL files before publishing")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--stage", default="deduped", choices=["raw", "validated", "deduped"])
    parser.add_argument("--signal", default=None)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    audit_stage(args.config, args.stage, signal=args.signal, fail_on_invalid=args.fail_on_invalid)


if __name__ == "__main__":
    cli()
