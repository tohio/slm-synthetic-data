from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_synth.cards import CARD_KINDS, build_dataset_card, write_dataset_card


@pytest.mark.parametrize("kind", sorted(CARD_KINDS))
def test_generation_dataset_cards_use_public_data_boundary(kind: str) -> None:
    card = build_dataset_card(kind)

    assert "path: data/*.jsonl" in card

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
def test_generation_dataset_cards_are_consumer_facing(kind: str) -> None:
    card = build_dataset_card(kind, total=10, family="example_family", teacher_model="teacher/model")

    for section in ["## Summary", "## Dataset", "## Schema", "## Intended Use", "## Limitations"]:
        assert section in card

    forbidden = [
        "## Run",
        "Generation run",
        "Teacher provider",
        "OpenRouter",
        "Planned",
        "Accepted",
        "Rejected rows",
        "Rejected pairs",
        "batch size",
        "concurrency",
        "cost",
        "data/sft/runs/",
        "data/dpo/runs/",
        "data/distillation/runs/",
        "data/distillation-dpo/runs/",
    ]
    for value in forbidden:
        assert value not in card


@pytest.mark.parametrize("kind", sorted(CARD_KINDS))
def test_generation_dataset_cards_include_kind_specific_schema(kind: str) -> None:
    card = build_dataset_card(kind)

    assert "# " in card
    assert "## Schema" in card

    if kind == "pretrain":
        assert '"text"' in card
    elif kind == "sft":
        assert '"messages"' in card
        assert '"assistant"' in card
    elif kind == "dpo":
        assert '"chosen"' in card
        assert '"rejected"' in card
        assert "Total pairs" in build_dataset_card(kind, total=3)
    elif kind == "distillation-sft":
        assert '"prompt"' in card
        assert '"response"' in card
        assert "Teacher model" in build_dataset_card(kind, teacher_model="teacher/model")
    elif kind == "distillation-dpo":
        assert '"chosen"' in card
        assert '"rejected"' in card
        assert "Teacher model" in build_dataset_card(kind, teacher_model="teacher/model")


def test_write_dataset_card_writes_run_readme_without_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sft-example"
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "manifests" / "sft-example.manifest.json").write_text(
        json.dumps({"generation_run": "sft-example", "total_rows": 12}) + "\n",
        encoding="utf-8",
    )

    readme = write_dataset_card("sft", run_dir)

    text = readme.read_text(encoding="utf-8")
    assert readme == run_dir / "README.md"
    assert "Total rows: `12`" in text
    assert "sft-example" not in text


def test_distillation_card_uses_teacher_model_but_not_provider(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "distill-example"
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "manifests" / "distill-example.manifest.json").write_text(
        json.dumps(
            {
                "generation_run": "distill-example",
                "teacher_model": "teacher/model",
                "teacher_provider": "openrouter",
                "total_rows": 10,
                "datasets": [{"signal": "arithmetic", "row_count": 10}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    text = write_dataset_card("distillation-sft", run_dir).read_text(encoding="utf-8")

    assert "Teacher model: `teacher/model`" in text
    assert "Signals: `arithmetic`" in text
    assert '"metadata": {' in text
    assert '"eval_family": "string | null"' in text
    assert "openrouter" not in text.lower()
    assert "distill-example" not in text
