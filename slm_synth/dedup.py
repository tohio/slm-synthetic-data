from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.record_quality import SIGNAL_FROM_FILE, canonical_exact_key, iter_jsonl, signal_to_filename


def _signals_from_args(src_dir: Path, signal: str | None) -> list[str]:
    if signal:
        return [signal]
    return [SIGNAL_FROM_FILE.get(path.name, path.stem) for path in sorted(src_dir.glob("*.jsonl"))]


def dedup_signal(src_dir: Path, dst_dir: Path, signal: str) -> tuple[int, int]:
    src = src_dir / signal_to_filename(signal)
    if not src.exists():
        print(f"[dedup] {src.name}: missing, skipped")
        return 0, 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    seen: set[str] = set()
    kept = 0
    dropped = 0
    with dst.open("w", encoding="utf-8") as out:
        for _, row, parse_issues in iter_jsonl(src):
            if parse_issues or row is None:
                dropped += 1
                continue
            key = canonical_exact_key(signal, row)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[dedup] {src.name}: kept={kept}, exact_dropped={dropped}, fuzzy_dropped=0")
    return kept, dropped


def run_from_config(config_path: str, signal: str | None = None) -> None:
    cfg = load_yaml_config(config_path)
    out = resolve_output_dir(cfg)
    src_dir = out / "validated"
    dst_dir = out / "deduped"
    print("[dedup] mode=exact enable_exact=True enable_fuzzy=False")

    total_kept = 0
    total_dropped = 0
    for sig in _signals_from_args(src_dir, signal):
        kept, dropped = dedup_signal(src_dir, dst_dir, sig)
        total_kept += kept
        total_dropped += dropped
    print(f"[dedup] Completed kept={total_kept}, exact_dropped={total_dropped}, fuzzy_dropped=0")


def run_positional(src_dir: str, dst_dir: str, signal: str | None = None) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    print("[dedup] mode=exact enable_exact=True enable_fuzzy=False")
    total_kept = 0
    total_dropped = 0
    for sig in _signals_from_args(src, signal):
        kept, dropped = dedup_signal(src, dst, sig)
        total_kept += kept
        total_dropped += dropped
    print(f"[dedup] Completed kept={total_kept}, exact_dropped={total_dropped}, fuzzy_dropped=0")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Exact-deduplicate validated synthetic JSONL records")
    parser.add_argument("src_dir", nargs="?", help="Legacy validated input directory")
    parser.add_argument("dst_dir", nargs="?", help="Legacy deduped output directory")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml")
    parser.add_argument("--signal", default=None, help="Optional signal filter")
    args = parser.parse_args()

    if args.config:
        run_from_config(args.config, signal=args.signal)
        return
    if not (args.src_dir and args.dst_dir):
        parser.error("Either --config or src_dir dst_dir is required")
    run_positional(args.src_dir, args.dst_dir, signal=args.signal)


if __name__ == "__main__":
    cli()
