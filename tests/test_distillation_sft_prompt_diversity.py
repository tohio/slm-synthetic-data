from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text, validate_prompt_preflight
from slm_synth.distillation_sft.push_hf import require_prompt_uniqueness_for_publish
from slm_synth.distillation_sft.seeds import build_seed_prompt_records
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS


def test_seed_smoke_prompt_records_are_unique_at_200_per_signal() -> None:
    for signal in sorted(DISTILLATION_SIGNALS):
        records = build_seed_prompt_records(signal=signal, count=200)
        normalized = [normalize_prompt_text(str(record["prompt"])) for record in records]
        assert len(normalized) == 200
        assert len(set(normalized)) == 200
        validate_prompt_preflight(records, require_unique_prompt_text=True)


def test_distillation_sft_publish_uniqueness_gate_rejects_duplicate_prompts(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": f"row-{index}", "prompt": "same prompt", "reasoning": None, "response": "ok"}
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="publish prompt uniqueness gate failed"):
        require_prompt_uniqueness_for_publish([path])


def test_distillation_sft_publish_uniqueness_gate_accepts_diverse_prompts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DISTILLATION_SFT_MIN_UNIQUE_PROMPTS", "0")
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": f"row-{index}", "prompt": f"prompt {index}", "reasoning": None, "response": "ok"}
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = require_prompt_uniqueness_for_publish([path])

    assert summary["row_count"] == 10
    assert summary["unique_prompt_count"] == 10
