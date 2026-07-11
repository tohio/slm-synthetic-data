from __future__ import annotations

from pathlib import Path

import pytest

from slm_synth.cards import CARD_KINDS, build_dataset_card, write_dataset_card


@pytest.mark.parametrize("kind", sorted(CARD_KINDS))
def test_generation_dataset_cards_use_public_data_boundary(kind: str) -> None:
    card = build_dataset_card(kind, run_id=f"{kind}-run")

    assert "path: data/*.jsonl" in card
    assert "artifacts/" in card
    assert "not part of the dataset split" in card

    forbidden_split_paths = [
        "path: artifacts/",
        "path: manifests/",
        "path: coverage.json",
        "path: scratch/",
        "path: batches/",
        "path: partial/",
        "path: rejected/",
        "path: retry/",
    ]
    for path in forbidden_split_paths:
        assert path not in card


@pytest.mark.parametrize("kind", sorted(CARD_KINDS))
def test_generation_dataset_cards_include_kind_specific_schema(kind: str) -> None:
    card = build_dataset_card(kind)

    assert "# " in card
    assert "## Schema" in card

    if kind == "pretrain":
        assert "`text`" in card
    elif kind == "sft":
        assert "`messages`" in card
        assert "assistant message" in card
    elif kind == "dpo":
        assert "`chosen`" in card
        assert "`rejected`" in card
    elif kind == "distillation-sft":
        assert "`prompt`" in card
        assert "`response`" in card
    elif kind == "distillation-dpo":
        assert "`chosen`" in card
        assert "`rejected`" in card


def test_write_dataset_card_writes_run_readme(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sft-example"
    readme = write_dataset_card("sft", run_dir)

    assert readme == run_dir / "README.md"
    assert readme.exists()
    assert "sft-example" in readme.read_text(encoding="utf-8")
