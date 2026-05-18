from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasketch import MinHash

from slm_synth.paths import validated_dir_from_config


def minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=128)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh


def dedup_file(path: Path, out_path: Path, threshold: float) -> tuple[int, int]:
    # This exact/fuzzy comparison is acceptable for the current smoke test.
    # Before 600M-token production runs, replace this O(n^2) scan with LSH.
    seen: dict[int, MinHash] = {}
    kept = 0
    dropped = 0

    with open(path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            obj = json.loads(line)
            text = json.dumps(obj, ensure_ascii=False)
            mh = minhash(text)

            dup = any(existing.jaccard(mh) >= threshold for existing in seen.values())
            if dup:
                dropped += 1
                continue

            seen[len(seen)] = mh
            out.write(line)
            kept += 1

    return kept, dropped


def main(validated_dir: str | Path) -> None:
    validated_dir = Path(validated_dir)
    deduped_dir = validated_dir.parent / "deduped"

    if not validated_dir.exists():
        raise FileNotFoundError(f"validated_dir does not exist: {validated_dir}")

    deduped_dir.mkdir(parents=True, exist_ok=True)
    for old in deduped_dir.glob("*.jsonl"):
        old.unlink()

    files = sorted(validated_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in validated_dir: {validated_dir}")

    total_kept = 0
    total_dropped = 0
    for file in files:
        out_file = deduped_dir / file.name
        kept, dropped = dedup_file(file, out_file, threshold=0.85)
        total_kept += kept
        total_dropped += dropped
        print(f"[dedup] {file.name}: kept={kept}, dropped={dropped}")

    print(f"[dedup] Completed kept={total_kept}, dropped={total_dropped}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate validated synthetic JSONL records.")
    parser.add_argument("validated_dir", nargs="?", help="Path to validated JSONL directory.")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml.")
    args = parser.parse_args()

    if args.config:
        validated_dir = validated_dir_from_config(args.config)
    elif args.validated_dir:
        validated_dir = Path(args.validated_dir)
    else:
        parser.error("provide either validated_dir or --config")

    main(validated_dir)


if __name__ == "__main__":
    cli()
