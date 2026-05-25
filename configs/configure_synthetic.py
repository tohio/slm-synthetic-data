#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import string
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from slm_synth.model_support import warn_if_unsupported_model

TEMPLATE_PATH = REPO_ROOT / "configs" / "synthetic_template.yaml"
OUTPUT_PATH = REPO_ROOT / "configs" / "synthetic.yaml"
load_dotenv()

PROFILES = {
    "speed": {"model": "deepseek/deepseek-v4-flash", "max_tokens": 10240, "temperature": 0.45, "top_p": 0.95, "concurrency": 8},
    "balanced": {"model": "deepseek/deepseek-v4-flash", "max_tokens": 10240, "temperature": 0.35, "top_p": 0.95, "concurrency": 4},
    "quality": {"model": "deepseek/deepseek-v4-flash", "max_tokens": 10240, "temperature": 0.25, "top_p": 0.95, "concurrency": 2},
}

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 1024


def generate_run_name(length: int = 7) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def resolve_hf_repo_from_dotenv() -> str:
    hf_username = os.getenv("HF_USERNAME", "").strip()
    hf_repo = os.getenv("HF_REPO", "").strip()
    if hf_repo and "/" in hf_repo:
        return hf_repo
    if hf_username and hf_repo:
        return f"{hf_username}/{hf_repo}"
    return "local/slm-synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded synthetic-data configuration")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--run", default=None)
    parser.add_argument("--hf_repo", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    # Backward-compatible accepted argument; OpenRouter grounded generation does not use Groq service tiers.
    parser.add_argument("--service-tier", default=None)
    args = parser.parse_args()
    if not MIN_BATCH_SIZE <= args.batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"--batch-size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE} "
            "for grounded throughput qualification"
        )

    preset = PROFILES[args.profile]
    model = args.model or preset["model"]
    warn_if_unsupported_model(model, context="configure")
    concurrency = args.concurrency or preset["concurrency"]
    if not MIN_CONCURRENCY <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(
            f"--concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY} "
            "for grounded throughput qualification"
        )
    max_tokens = args.max_tokens or preset["max_tokens"]
    if max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    run_name = args.run or generate_run_name()
    hf_repo = args.hf_repo or resolve_hf_repo_from_dotenv()
    filled = (
        TEMPLATE_PATH.read_text()
        .replace("__RUN_NAME__", run_name)
        .replace("__OUTPUT_DIR__", f"${{DATA_DIR}}/{run_name}")
        .replace("__TARGET_TOKENS__", str(args.tokens))
        .replace("__MODEL__", model)
        .replace("__MAX_TOKENS__", str(max_tokens))
        .replace("__TEMPERATURE__", str(preset["temperature"]))
        .replace("__TOP_P__", str(preset["top_p"]))
        .replace("__BATCH_SIZE__", str(args.batch_size))
        .replace("__CONCURRENCY__", str(concurrency))
        .replace("__HF_REPO__", hf_repo)
    )
    OUTPUT_PATH.write_text(filled)
    print(f"Generated grounded config at {OUTPUT_PATH}")
    print(f"run_name={run_name}")
    print(
        f"model={model} batch_size={args.batch_size} concurrency={concurrency} "
        f"max_tokens={max_tokens} target_total_tokens={args.tokens}"
    )
    print(f"hf_repo={hf_repo}")


if __name__ == "__main__":
    main()
