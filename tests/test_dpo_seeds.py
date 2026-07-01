import pytest

from slm_synth.dpo.seeds import (
    build_answer_only_arithmetic_rows,
    build_seed_rows,
)
from slm_synth.taxonomy.holdouts import HoldoutRegistry


class RejectFirstAdditionRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {"type": "arithmetic", "operation": "add", "operands": [3, 4]}:
            raise ValueError("candidate holdout_key matches eval holdout key")


def test_build_answer_only_arithmetic_rows_creates_valid_dpo_rows():
    rows = build_answer_only_arithmetic_rows(count=2)

    assert [row["id"] for row in rows] == [
        "dpo_answer_only_arithmetic_000001",
        "dpo_answer_only_arithmetic_000002",
    ]
    assert all(row["prompt"][0]["role"] == "user" for row in rows)
    assert all(row["chosen"][0]["role"] == "assistant" for row in rows)
    assert all(row["rejected"][0]["role"] == "assistant" for row in rows)
    assert rows[0]["chosen"][0]["content"] == "7"
    assert rows[0]["rejected"][0]["content"] == "The answer is 7 because 3 plus 4 equals 7."
    assert rows[0]["metadata"]["failure_mode"] == "extra_explanation"
    assert rows[0]["metadata"]["category"] == "answer_only_compliance"
    assert rows[0]["metadata"]["eval_family"] == "basic_arithmetic_qa"


def test_build_seed_rows_dispatches_supported_family():
    rows = build_seed_rows(family="answer_only_arithmetic", count=1, start_index=7)

    assert rows[0]["id"] == "dpo_answer_only_arithmetic_000007"


def test_build_seed_rows_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported DPO seed family"):
        build_seed_rows(family="unknown", count=1)


def test_build_answer_only_arithmetic_rows_rejects_non_positive_count():
    with pytest.raises(ValueError, match="positive integer"):
        build_answer_only_arithmetic_rows(count=0)


def test_build_answer_only_arithmetic_rows_skips_holdout_keys():
    registry = RejectFirstAdditionRegistry()

    rows = build_answer_only_arithmetic_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "arithmetic", "operation": "add", "operands": [3, 4]}
    assert rows[0]["prompt"][0]["content"] != "Answer with only the number: What is 3 + 4?"
    assert rows[0]["id"] == "dpo_answer_only_arithmetic_000001"


def test_build_answer_only_arithmetic_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "basic_arithmetic_qa": [
                {
                    "id": "factual_math_basic",
                    "prompt": "What is 2 + 2?",
                    "answer": "4",
                    "holdout_key": {"type": "arithmetic", "operation": "add", "operands": [2, 2]},
                }
            ]
        }
    )

    rows = build_answer_only_arithmetic_rows(count=1, holdout_registry=registry)

    assert "2 + 2" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["chosen"][0]["content"] == "7"
    assert rows[0]["chosen"] != rows[0]["rejected"]
