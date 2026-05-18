from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config after reading .env, returning an empty dict for empty YAML."""
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_output_dir(cfg: Mapping[str, Any]) -> Path:
    """Resolve cfg['output_dir'] consistently across all pipeline stages.

    The generated config commonly stores output_dir as '${DATA_DIR}/<run_id>'.
    In an interactive shell DATA_DIR may not be exported, so we default DATA_DIR
    to 'data/runs'. This matches bootstrap/generate behavior and prevents
    validate/dedup/push from looking for a literal '${DATA_DIR}' directory.
    """
    raw = str(cfg.get("output_dir", "")).strip()
    if not raw:
        raise ValueError("configs/synthetic.yaml is missing required key: output_dir")

    os.environ.setdefault("DATA_DIR", "data/runs")
    resolved = os.path.expandvars(raw)
    return Path(resolved)


def raw_dir_from_config(path: str | Path) -> Path:
    return resolve_output_dir(load_yaml_config(path)) / "raw"


def validated_dir_from_config(path: str | Path) -> Path:
    return resolve_output_dir(load_yaml_config(path)) / "validated"


def deduped_dir_from_config(path: str | Path) -> Path:
    return resolve_output_dir(load_yaml_config(path)) / "deduped"


def repo_id_from_config_or_env(cfg: Mapping[str, Any]) -> str:
    hf_cfg = cfg.get("hf") or cfg.get("huggingface") or {}
    repo = hf_cfg.get("repo_id") or os.getenv("HF_REPO_ID")
    if repo:
        return str(repo)

    username = os.getenv("HF_USERNAME")
    repo_name = os.getenv("HF_REPO")
    if username and repo_name:
        return f"{username}/{repo_name}"

    raise ValueError(
        "Missing HF repo. Set HF_REPO_ID or HF_USERNAME+HF_REPO, "
        "or add hf.repo_id to config."
    )
