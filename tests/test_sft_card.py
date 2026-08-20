from pathlib import Path

import pytest

from slm_synth.cards import build_dataset_card
from slm_synth.sft.card import load_sft_dataset_card_configs, require_sft_dataset_card_configs


FAMILIES = ["creative_writing", "grounded_qa_and_reading"]


def _write_consolidated_card(tmp_path: Path) -> Path:
    readme = tmp_path / "README.md"
    readme.write_text(
        build_dataset_card("sft", total=2, signals=FAMILIES),
        encoding="utf-8",
    )
    return readme


def _resolve_config_paths(root: Path, files: list[dict[str, str]]) -> list[Path]:
    resolved: list[Path] = []
    for item in files:
        assert item["split"] == "train"
        resolved.extend(sorted(root.glob(item["path"])))
    return resolved


def test_consolidated_sft_card_has_default_and_per_family_configs(tmp_path):
    readme = _write_consolidated_card(tmp_path)

    configs = require_sft_dataset_card_configs(readme, families=FAMILIES)

    assert list(configs) == ["default", *FAMILIES]
    assert configs["default"] == [{"split": "train", "path": "data/*.jsonl"}]
    assert configs["creative_writing"] == [
        {"split": "train", "path": "data/creative_writing.jsonl"}
    ]
    assert configs["grounded_qa_and_reading"] == [
        {"split": "train", "path": "data/grounded_qa_and_reading.jsonl"}
    ]


def test_consolidated_sft_default_and_family_configs_reference_same_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for family in FAMILIES:
        (data_dir / f"{family}.jsonl").write_text("{}\n", encoding="utf-8")
    configs = load_sft_dataset_card_configs(_write_consolidated_card(tmp_path))

    default_files = _resolve_config_paths(tmp_path, configs["default"])
    family_files = [
        path
        for family in FAMILIES
        for path in _resolve_config_paths(tmp_path, configs[family])
    ]

    assert default_files == [data_dir / f"{family}.jsonl" for family in FAMILIES]
    assert family_files == default_files


def test_consolidated_sft_card_rejects_missing_family_config(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/*.jsonl\n---\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match run families"):
        require_sft_dataset_card_configs(readme, families=FAMILIES)
