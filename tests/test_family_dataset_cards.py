from __future__ import annotations

import json
from pathlib import Path

from slm_synth.family_cards import build_family_dataset_card


def test_build_sft_family_dataset_card_uses_family_count(tmp_path: Path) -> None:
    path = tmp_path / "function_completion_body_only.jsonl"
    rows = [
        {
            "id": "sft_1",
            "messages": [
                {"role": "user", "content": "Complete the function."},
                {"role": "assistant", "content": "return a + b"},
            ],
            "metadata": {
                "category": "code_generation",
                "difficulty": 2,
                "template_family": "python_function_body_only",
                "eval_family": "function_completion_body_only",
            },
        },
        {
            "id": "sft_2",
            "messages": [
                {"role": "user", "content": "Complete the function."},
                {"role": "assistant", "content": "return x"},
            ],
            "metadata": {
                "category": "code_generation",
                "difficulty": 2,
                "template_family": "python_function_body_only",
                "eval_family": "function_completion_body_only",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    card = build_family_dataset_card(
        kind="sft",
        family="function_completion_body_only",
        jsonl_paths=[path],
    )

    assert "# SLM Synthetic SFT — function_completion_body_only" in card
    assert "- Total rows: `2`" in card
    assert "- Signal: `function_completion_body_only`" in card
    assert "ai_concept_explanation" not in card
    assert "400000" not in card


def test_build_dpo_family_dataset_card_uses_family_count(tmp_path: Path) -> None:
    path = tmp_path / "code_explanation_no_code.jsonl"
    row = {
        "id": "dpo_1",
        "prompt": [{"role": "user", "content": "Explain this code without code."}],
        "chosen": [{"role": "assistant", "content": "It adds two values."}],
        "rejected": [{"role": "assistant", "content": "The code is `a + b`."}],
        "metadata": {
            "category": "general_instruction_following",
            "difficulty": 2,
            "template_family": "code_explanation_plain_text",
            "eval_family": "code_explanation_no_code",
            "failure_mode": "code_includes_explanation",
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    card = build_family_dataset_card(
        kind="dpo",
        family="code_explanation_no_code",
        jsonl_paths=[path],
    )

    assert "# SLM Synthetic DPO — code_explanation_no_code" in card
    assert "- Total pairs: `1`" in card
    assert "- Signal: `code_explanation_no_code`" in card
    assert "basic_arithmetic_qa" not in card
    assert "90000" not in card
