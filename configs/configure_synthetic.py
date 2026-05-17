#!/usr/bin/env python3
import argparse
import os
import re
import requests
import yaml
from pathlib import Path
from dotenv import load_dotenv


DEFAULT_MODEL = "openai/gpt-oss-20b"


# -----------------------------
# 1. Fetch models from Groq API
# -----------------------------
def fetch_groq_models():
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment or .env file.")

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


# -----------------------------
# 2. Infer model size from name
# -----------------------------
def infer_model_size(model_name: str) -> float:
    """
    Extracts model size in billions from patterns like:
    - 7b, 8b, 17b, 20b, 32b, 70b, 120b
    - 22m, 86m → treated as tiny → 0.1B
    """

    # Look for "<number>b"
    b_match = re.search(r"(\d+)\s*b", model_name.lower())
    if b_match:
        return float(b_match.group(1))

    # Look for "<number>m"
    m_match = re.search(r"(\d+)\s*m", model_name.lower())
    if m_match:
        return float(m_match.group(1)) / 1000.0  # convert to billions

    # Fallback: treat unknown sizes as small
    return 8.0


# -----------------------------
# 3. Bucket → profile mapping
# -----------------------------
def select_profile(size_b: float):
    """
    Option A: nearest-bucket mapping
    """
    if size_b <= 10:
        return {
            "batch_size": 16,
            "max_tokens": 384,
            "temperature": 0.3,
            "top_p": 0.9,
            "concurrency": 4,
        }

    if size_b <= 30:
        return {
            "batch_size": 32,
            "max_tokens": 512,
            "temperature": 0.4,
            "top_p": 0.95,
            "concurrency": 2,
        }

    if size_b <= 80:
        return {
            "batch_size": 64,
            "max_tokens": 768,
            "temperature": 0.2,
            "top_p": 0.95,
            "concurrency": 1,
        }

    # Extra-large fallback
    print(f"Warning: Model size {size_b}B is very large. Using 70B profile.")
    return {
        "batch_size": 64,
        "max_tokens": 768,
        "temperature": 0.2,
        "top_p": 0.95,
        "concurrency": 1,
    }


# -----------------------------
# 4. Patch existing YAML config
# -----------------------------
def update_config(config_path, model_name, tokens, batch_override=None):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())

    # Infer size
    size_b = infer_model_size(model_name)
    profile = select_profile(size_b)

    # Patch fields
    cfg["target_total_tokens"] = tokens

    cfg["backend"]["provider"] = "groq"
    cfg["backend"]["model"] = model_name
    cfg["backend"]["max_tokens"] = profile["max_tokens"]
    cfg["backend"]["temperature"] = profile["temperature"]
    cfg["backend"]["top_p"] = profile["top_p"]
    cfg["backend"]["parallel_requests"] = profile["concurrency"]

    cfg["rate_limit"]["max_concurrency"] = profile["concurrency"]

    cfg["generation"]["batch_size"] = batch_override or profile["batch_size"]

    # Write back
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"Updated config written to {cfg_path}")


# -----------------------------
# 5. CLI entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Modify existing synthetic.yaml using Groq model profiles")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()

    # Fetch models from Groq
    available = fetch_groq_models()

    if args.model not in available:
        print(f"Model '{args.model}' not found in Groq model list.")
        print("Available Groq models:")
        for m in available:
            print(" -", m)
        raise ValueError(f"Model '{args.model}' not found.")

    update_config(
        config_path=args.config,
        model_name=args.model,
        tokens=args.tokens,
        batch_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
