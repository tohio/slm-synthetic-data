"""Dataset-card configuration validation for consolidated generic DPO publication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml


def load_dpo_dataset_card_configs(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load and normalize Hugging Face data-file configurations from README YAML."""
    readme_path = Path(path)
    text = readme_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"DPO dataset card is missing YAML front matter: {readme_path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"DPO dataset card has unterminated YAML front matter: {readme_path}")
    value = yaml.safe_load(text[4:end])
    if not isinstance(value, Mapping) or not isinstance(value.get("configs"), list):
        raise ValueError("DPO dataset card YAML must contain a configs list")

    configs: dict[str, list[dict[str, str]]] = {}
    for raw_config in value["configs"]:
        if not isinstance(raw_config, Mapping):
            raise ValueError("DPO dataset card config must be an object")
        name = raw_config.get("config_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("DPO dataset card config_name must be a non-empty string")
        if name in configs:
            raise ValueError(f"duplicate DPO dataset card config: {name}")
        raw_files = raw_config.get("data_files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"DPO dataset card config {name} must contain data_files")
        files: list[dict[str, str]] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise ValueError(f"DPO dataset card config {name} data file must be an object")
            split = raw_file.get("split")
            data_path = raw_file.get("path")
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"DPO dataset card config {name} split must be non-empty")
            if not isinstance(data_path, str) or not data_path.strip():
                raise ValueError(f"DPO dataset card config {name} path must be non-empty")
            files.append({"split": split, "path": data_path})
        configs[name] = files
    return configs


def require_dpo_dataset_card_configs(
    path: str | Path, *, preference_dimensions: Iterable[str]
) -> dict[str, list[dict[str, str]]]:
    """Require one all-dimension default config and one config per dimension."""
    expected_dimensions = tuple(sorted(set(preference_dimensions)))
    if not expected_dimensions:
        raise ValueError(
            "at least one DPO preference dimension is required for dataset-card configs"
        )
    configs = load_dpo_dataset_card_configs(path)
    expected_names = {"default", *expected_dimensions}
    if set(configs) != expected_names:
        raise ValueError(
            "DPO dataset card configs do not match run preference dimensions: "
            f"expected {sorted(expected_names)}, got {sorted(configs)}"
        )
    if configs["default"] != [{"split": "train", "path": "data/*.jsonl"}]:
        raise ValueError(
            "DPO default config must load all preference-dimension JSONL files as train"
        )
    for dimension in expected_dimensions:
        expected = [{"split": "train", "path": f"data/{dimension}.jsonl"}]
        if configs[dimension] != expected:
            raise ValueError(
                f"DPO preference-dimension config {dimension} must load only "
                f"data/{dimension}.jsonl"
            )
    return configs
