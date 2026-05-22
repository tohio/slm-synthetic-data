#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import random
import string

import requests
from dotenv import load_dotenv
from slm_synth.model_support import warn_if_unsupported_model

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

TEMPLATE_PATH = ROOT / "configs" / "synthetic_template.yaml"
OUTPUT_PATH = ROOT / "configs" / "synthetic.yaml"

load_dotenv()

# ---------------------------------------------------------
# Profile presets
# ---------------------------------------------------------
# Both candidate generation and independent response completion use the
# selected profile model. Balanced/speed default to 8B for economical bulk
# generation; quality remains available for explicit 70B comparisons.
PROFILES = {
    "speed": {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 1024,
        "temperature": 0.55,
        "top_p": 0.95,
        "concurrency": 12,
        "avg_tokens_per_sample": 100,
        "service_tier": "flex",
    },
    "balanced": {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 1536,
        "temperature": 0.45,
        "top_p": 0.95,
        "concurrency": 8,
        "avg_tokens_per_sample": 110,
        "service_tier": "flex",
    },
    "quality": {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 2048,
        "temperature": 0.35,
        "top_p": 0.95,
        "concurrency": 4,
        "avg_tokens_per_sample": 130,
        "service_tier": "flex",
    },
}

SIGNAL_DEFAULTS = {
    "arithmetic": {"batch_size": 8, "avg_tokens_per_sample": 60},
    "task_code": {"batch_size": 2, "avg_tokens_per_sample": 160},
    "educational_qa_mcq_math": {"batch_size": 8, "avg_tokens_per_sample": 100},
    "educational_qa_mcq_general": {"batch_size": 8, "avg_tokens_per_sample": 100},
    "factual_restraint": {"batch_size": 4, "avg_tokens_per_sample": 90},
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def generate_run_name(length=7):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def fetch_groq_models():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def scaled_max_tokens(batch_size: int, avg_tokens_per_sample: int, floor: int) -> int:
    # JSON syntax, prompts that produce code, and safe-answer text need headroom.
    return max(int(floor), int(batch_size * avg_tokens_per_sample * 4))


def sanitize_service_tier(value: str) -> str:
    value = (value or "flex").strip().lower()
    if value not in {"auto", "default", "flex"}:
        raise ValueError("--service-tier must be one of: auto, default, flex")
    return value




def resolve_hf_repo_from_dotenv() -> str:
    """Resolve the base Hugging Face dataset repo from .env.

    Supported .env forms:
      HF_USERNAME=tohio
      HF_REPO=slm-synthetic

    or:
      HF_REPO=tohio/slm-synthetic

    The generated config stores the fully-qualified repo id, e.g.
      tohio/slm-synthetic
    """
    load_dotenv()
    hf_username = os.getenv("HF_USERNAME", "").strip()
    hf_repo = os.getenv("HF_REPO", "").strip()

    if hf_repo and "/" in hf_repo:
        return hf_repo

    if hf_username and hf_repo:
        return f"{hf_username}/{hf_repo}"

    if hf_repo:
        raise ValueError(
            "HF_REPO is set without HF_USERNAME. Set HF_USERNAME in .env "
            "or set HF_REPO to a fully-qualified repo id like 'tohio/slm-synthetic'."
        )

    raise ValueError(
        "Missing Hugging Face repo settings. Set HF_USERNAME and HF_REPO in .env, "
        "or set HF_REPO to a fully-qualified repo id like 'tohio/slm-synthetic'."
    )

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified synthetic config generator")
    parser.add_argument("--profile", choices=["speed", "balanced", "quality"], default="balanced")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--run", default=None)
    parser.add_argument("--hf_repo", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--service-tier", default=None, choices=["auto", "default", "flex"])
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    resolved_hf_repo = args.hf_repo or resolve_hf_repo_from_dotenv()

    preset = PROFILES[args.profile]

    run_name = args.run if args.run else generate_run_name()
    model_name = args.model if args.model else preset["model"]
    warn_if_unsupported_model(model_name, context="configure")
    concurrency = int(args.concurrency or preset["concurrency"])
    service_tier = sanitize_service_tier(args.service_tier or preset.get("service_tier", "flex"))

    if not args.skip_model_check:
        available = fetch_groq_models()
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' not found in Groq model list.")

    global_batch_size = int(args.batch_size or SIGNAL_DEFAULTS["arithmetic"]["batch_size"])

    signal_values = {}
    for signal, defaults in SIGNAL_DEFAULTS.items():
        if args.batch_size is not None:
            batch_size = max(1, int(args.batch_size))
        else:
            batch_size = int(defaults["batch_size"])
        avg_tokens = int(defaults["avg_tokens_per_sample"])
        max_tokens = scaled_max_tokens(batch_size, avg_tokens, int(preset["max_tokens"]))
        signal_values[signal] = {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
        }

    template = TEMPLATE_PATH.read_text()

    filled = (
        template
        .replace("__RUN_NAME__", run_name)
        .replace("__OUTPUT_DIR__", f"${{DATA_DIR}}/{run_name}")
        .replace("__TARGET_TOKENS__", str(args.tokens))
        .replace("__MODEL__", model_name)
        .replace("__MAX_TOKENS__", str(preset["max_tokens"]))
        .replace("__TEMPERATURE__", str(preset["temperature"]))
        .replace("__TOP_P__", str(preset["top_p"]))
        .replace("__CONCURRENCY__", str(concurrency))
        .replace("__SERVICE_TIER__", service_tier)
        .replace("__AVG_TOKENS_PER_SAMPLE__", str(preset["avg_tokens_per_sample"]))
        .replace("__BATCH_SIZE__", str(global_batch_size))
        .replace("__HF_REPO__", resolved_hf_repo)
    )

    replacements = {
        "__ARITHMETIC_BATCH_SIZE__": signal_values["arithmetic"]["batch_size"],
        "__ARITHMETIC_MAX_TOKENS__": signal_values["arithmetic"]["max_tokens"],
        "__TASK_CODE_BATCH_SIZE__": signal_values["task_code"]["batch_size"],
        "__TASK_CODE_MAX_TOKENS__": signal_values["task_code"]["max_tokens"],
        "__EDUCATIONAL_QA_MCQ_MATH_BATCH_SIZE__": signal_values["educational_qa_mcq_math"]["batch_size"],
        "__EDUCATIONAL_QA_MCQ_MATH_MAX_TOKENS__": signal_values["educational_qa_mcq_math"]["max_tokens"],
        "__EDUCATIONAL_QA_MCQ_GENERAL_BATCH_SIZE__": signal_values["educational_qa_mcq_general"]["batch_size"],
        "__EDUCATIONAL_QA_MCQ_GENERAL_MAX_TOKENS__": signal_values["educational_qa_mcq_general"]["max_tokens"],
        "__FACTUAL_RESTRAINT_BATCH_SIZE__": signal_values["factual_restraint"]["batch_size"],
        "__FACTUAL_RESTRAINT_MAX_TOKENS__": signal_values["factual_restraint"]["max_tokens"],
    }
    for key, value in replacements.items():
        filled = filled.replace(key, str(value))

    OUTPUT_PATH.write_text(filled)
    print(f"Generated unified config at {OUTPUT_PATH}")
    print(f"run_name={run_name}")
    print(f"model={model_name} service_tier={service_tier} concurrency={concurrency}")
    print(f"hf_repo={resolved_hf_repo}")


if __name__ == "__main__":
    main()
