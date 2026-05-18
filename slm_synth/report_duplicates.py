
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from slm_synth.paths import load_yaml_config, resolve_output_dir


def canonical_record(line: str) -> str:
    obj = json.loads(line)
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def record_digest(line: str) -> str:
    return hashlib.sha256(canonical_record(line).encode("utf-8")).hexdigest()


def scan_file(path: Path) -> Tuple[int, int, int]:
    total = 0
    bad = 0
    seen = set()
    duplicates = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                digest = record_digest(line)
            except Exception:
                bad += 1
                continue
            if digest in seen:
                duplicates += 1
            else:
                seen.add(digest)
    return total, duplicates, bad


def main(config: str, stage: str) -> None:
    cfg = load_yaml_config(config)
    out = resolve_output_dir(cfg)
    stage_dir = out / stage
    if not stage_dir.exists():
        raise SystemExit(f"Stage directory does not exist: {stage_dir}")

    total_all = 0
    dup_all = 0
    bad_all = 0
    for path in sorted(stage_dir.glob("*.jsonl")):
        total, dup, bad = scan_file(path)
        total_all += total
        dup_all += dup
        bad_all += bad
        unique = total - dup - bad
        pct = (dup / total * 100.0) if total else 0.0
        print(
            f"[duplicates] {stage}/{path.name}: total={total}, unique={unique}, "
            f"duplicates={dup} ({pct:.2f}%), bad_json={bad}"
        )
    pct_all = (dup_all / total_all * 100.0) if total_all else 0.0
    print(
        f"[duplicates] Completed stage={stage} total={total_all}, "
        f"duplicates={dup_all} ({pct_all:.2f}%), bad_json={bad_all}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report exact duplicate rates for a generated stage.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--stage", default="raw", choices=["raw", "validated", "deduped"])
    args = parser.parse_args()
    main(args.config, args.stage)
