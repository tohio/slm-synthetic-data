"""Dataset-card configuration validation for consolidated generic SFT publication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def load_sft_dataset_card_configs(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load and normalize Hugging Face data-file configurations from README YAML."""
    readme_path = Path(path)
    text = readme_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"SFT dataset card is missing YAML front matter: {readme_path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"SFT dataset card has unterminated YAML front matter: {readme_path}")
    value = yaml.safe_load(text[4:end])
    if not isinstance(value, Mapping) or not isinstance(value.get("configs"), list):
        raise ValueError("SFT dataset card YAML must contain a configs list")

    configs: dict[str, list[dict[str, str]]] = {}
    for raw_config in value["configs"]:
        if not isinstance(raw_config, Mapping):
            raise ValueError("SFT dataset card config must be an object")
        name = raw_config.get("config_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("SFT dataset card config_name must be a non-empty string")
        if name in configs:
            raise ValueError(f"duplicate SFT dataset card config: {name}")
        raw_files = raw_config.get("data_files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"SFT dataset card config {name} must contain data_files")
        files: list[dict[str, str]] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise ValueError(f"SFT dataset card config {name} data file must be an object")
            split = raw_file.get("split")
            data_path = raw_file.get("path")
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"SFT dataset card config {name} split must be non-empty")
            if not isinstance(data_path, str) or not data_path.strip():
                raise ValueError(f"SFT dataset card config {name} path must be non-empty")
            files.append({"split": split, "path": data_path})
        configs[name] = files
    return configs


def require_sft_dataset_card_configs(path: str | Path, *, families: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """Require one all-family default config and one exact config per family."""
    expected_families = tuple(sorted(set(families)))
    if not expected_families:
        raise ValueError("at least one SFT family is required for dataset-card configs")
    configs = load_sft_dataset_card_configs(path)
    expected_names = {"default", *expected_families}
    if set(configs) != expected_names:
        raise ValueError(
            "SFT dataset card configs do not match run families: "
            f"expected {sorted(expected_names)}, got {sorted(configs)}"
        )
    if configs["default"] != [{"split": "train", "path": "data/*.jsonl"}]:
        raise ValueError("SFT default config must load all family JSONL files as train")
    for family in expected_families:
        expected = [{"split": "train", "path": f"data/{family}.jsonl"}]
        if configs[family] != expected:
            raise ValueError(f"SFT family config {family} must load only data/{family}.jsonl")
    return configs
