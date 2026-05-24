#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from slm_synth.paths import load_yaml_config, resolve_output_dir


def estimated_tokens(record: dict, chars_per_token: float) -> int:
    text = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return max(1, math.ceil(len(text) / chars_per_token))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return sorted(values)[index]


def main(config: str, stage: str, chars_per_token: float) -> None:
    cfg = load_yaml_config(config)
    stage_dir = resolve_output_dir(cfg) / stage
    reports = {}
    for path in sorted(stage_dir.glob("*.jsonl")):
        lengths = [estimated_tokens(json.loads(line), chars_per_token) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lengths:
            continue
        reports[path.stem] = {
            "records": len(lengths),
            "mean_estimated_tokens": round(mean(lengths), 2),
            "median_estimated_tokens": median(lengths),
            "p90_estimated_tokens": percentile(lengths, 0.90),
            "recommended_avg_tokens_per_sample": math.ceil(mean(lengths)),
        }
        row = reports[path.stem]
        print(
            f"[lengths] {path.stem}: records={row['records']}, mean_estimated_tokens={row['mean_estimated_tokens']}, "
            f"median={row['median_estimated_tokens']}, p90={row['p90_estimated_tokens']}, "
            f"recommended_avg_tokens_per_sample={row['recommended_avg_tokens_per_sample']}"
        )
    output_path = resolve_output_dir(cfg) / "manifests" / f"length_report_{stage}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"stage": stage, "chars_per_token": chars_per_token, "signals": reports}, indent=2), encoding="utf-8")
    print(f"[lengths] Saved report: {output_path}")
    print("[lengths] Estimates use serialized characters/chars_per_token; no downstream tokenizer is imported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate per-record length for target-row calibration.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--stage", default="deduped", choices=["raw", "validated", "deduped"])
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    args = parser.parse_args()
    main(args.config, args.stage, args.chars_per_token)
