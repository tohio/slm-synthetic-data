#!/usr/bin/env python3
import argparse
import os
import random
import string
import requests
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "configs" / "synthetic_template.yaml"
OUTPUT_PATH = ROOT / "configs" / "synthetic.yaml"

load_dotenv()

# ---------------------------------------------------------
# Profile presets
# ---------------------------------------------------------
PROFILES = {
    "speed": {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 192,
        "temperature": 0.4,
        "top_p": 0.9,
        "concurrency": 12,
        "avg_tokens_per_sample": 60,
    },
    "balanced": {
        "model": "openai/gpt-oss-20b",
        "max_tokens": 256,
        "temperature": 0.4,
        "top_p": 0.95,
        "concurrency": 8,
        "avg_tokens_per_sample": 80,
    },
    "quality": {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 384,
        "temperature": 0.2,
        "top_p": 0.95,
        "concurrency": 4,
        "avg_tokens_per_sample": 120,
    },
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def generate_run_name(length=7):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def fetch_groq_models():
    # Always load .env from project root
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def compute_samples_per_signal(target_tokens: int, avg_tokens_per_sample: int):
    return max(1, target_tokens // avg_tokens_per_sample)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified synthetic config generator")
    parser.add_argument("--profile", choices=["speed", "balanced", "quality"], default="balanced")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--run", default=None)
    parser.add_argument("--hf_repo", default="user/repo")
    args = parser.parse_args()

    preset = PROFILES[args.profile]

    # Determine run name
    run_name = args.run if args.run else generate_run_name()

    # Determine model
    model_name = args.model if args.model else preset["model"]

    # Validate model exists
    available = fetch_groq_models()
    if model_name not in available:
        raise ValueError(f"Model '{model_name}' not found in Groq model list.")

    # Compute samples_per_signal
    samples = compute_samples_per_signal(
        args.tokens,
        avg_tokens_per_sample=preset["avg_tokens_per_sample"]
    )

    # Load template
    template = TEMPLATE_PATH.read_text()

    # Fill template
    filled = (
        template
        .replace("__RUN_NAME__", run_name)
        .replace("__OUTPUT_DIR__", f"${{DATA_DIR}}/{run_name}")
        .replace("__TARGET_TOKENS__", str(args.tokens))
        .replace("__MODEL__", model_name)
        .replace("__MAX_TOKENS__", str(preset["max_tokens"]))
        .replace("__TEMPERATURE__", str(preset["temperature"]))
        .replace("__TOP_P__", str(preset["top_p"]))
        .replace("__CONCURRENCY__", str(preset["concurrency"]))
        .replace("__SAMPLES_PER_SIGNAL__", str(samples))
        .replace("__HF_REPO__", args.hf_repo)
    )

    # Write output
    OUTPUT_PATH.write_text(filled)
    print(f"Generated unified config at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
