import pytest

from slm_synth.dpo.seeds import (
    build_answer_only_arithmetic_rows,
    build_list_exact_n_items_rows,
    build_private_or_unverifiable_company_fact_rows,
    build_repeat_exact_n_times_rows,
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


class RejectFirstRepeatRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {"type": "repeat_exact_n_times", "token": "dog", "count": 2}:
            raise ValueError("candidate holdout_key matches eval holdout key")


class RejectFirstListRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {"type": "list_exact_n_items", "item_type": "animals", "count": 2}:
            raise ValueError("candidate holdout_key matches eval holdout key")


class RejectFirstPrivateCompanyRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {
            "type": "unverifiable_company_fact",
            "company": "Cohere",
            "fact": "private_revenue",
            "time_window": "last_week",
        }:
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


def test_build_seed_rows_dispatches_repeat_family():
    rows = build_seed_rows(family="repeat_exact_n_times", count=1, start_index=7)

    assert rows[0]["id"] == "dpo_repeat_exact_n_times_000007"


def test_build_seed_rows_dispatches_list_family():
    rows = build_seed_rows(family="list_exact_n_items", count=1, start_index=7)

    assert rows[0]["id"] == "dpo_list_exact_n_items_000007"


def test_build_seed_rows_dispatches_private_company_family():
    rows = build_seed_rows(family="private_or_unverifiable_company_fact", count=1, start_index=7)

    assert rows[0]["id"] == "dpo_private_or_unverifiable_company_fact_000007"


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


def test_build_repeat_exact_n_times_rows_creates_valid_dpo_rows():
    rows = build_repeat_exact_n_times_rows(count=2)

    assert [row["id"] for row in rows] == [
        "dpo_repeat_exact_n_times_000001",
        "dpo_repeat_exact_n_times_000002",
    ]
    assert rows[0]["prompt"][0]["content"] == "Repeat dog exactly two times."
    assert rows[0]["chosen"][0]["content"] == "dog dog"
    assert rows[0]["rejected"][0]["content"] == "dog dog dog"
    assert rows[0]["metadata"]["category"] == "exact_output_format_control"
    assert rows[0]["metadata"]["failure_mode"] == "format_violation"
    assert rows[0]["metadata"]["template_family"] == "repeat_word_count"
    assert rows[0]["metadata"]["eval_family"] == "repeat_exact_n_times"


def test_build_repeat_exact_n_times_rows_skips_holdout_keys():
    registry = RejectFirstRepeatRegistry()

    rows = build_repeat_exact_n_times_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "repeat_exact_n_times", "token": "dog", "count": 2}
    assert rows[0]["prompt"][0]["content"] != "Repeat dog exactly two times."
    assert rows[0]["id"] == "dpo_repeat_exact_n_times_000001"


def test_build_repeat_exact_n_times_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "repeat_exact_n_times": [
                {
                    "id": "format_repeat_cat",
                    "prompt": "Repeat cat exactly three times.",
                    "answer": "cat cat cat",
                    "holdout_key": {"type": "repeat_exact_n_times", "token": "cat", "count": 3},
                }
            ]
        }
    )

    rows = build_repeat_exact_n_times_rows(count=1, holdout_registry=registry)

    assert "cat" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["chosen"][0]["content"] == "dog dog"
    assert rows[0]["chosen"] != rows[0]["rejected"]


def test_build_list_exact_n_items_rows_creates_valid_dpo_rows():
    rows = build_list_exact_n_items_rows(count=2)

    assert [row["id"] for row in rows] == [
        "dpo_list_exact_n_items_000001",
        "dpo_list_exact_n_items_000002",
    ]
    assert rows[0]["prompt"][0]["content"] == "List exactly two animals."
    assert rows[0]["chosen"][0]["content"] == "dog, cat"
    assert rows[0]["rejected"][0]["content"] == "dog, cat, horse"
    assert rows[0]["metadata"]["category"] == "exact_output_format_control"
    assert rows[0]["metadata"]["failure_mode"] == "format_violation"
    assert rows[0]["metadata"]["template_family"] == "list_n_items"
    assert rows[0]["metadata"]["eval_family"] == "list_exact_n_items"


def test_build_list_exact_n_items_rows_skips_holdout_keys():
    registry = RejectFirstListRegistry()

    rows = build_list_exact_n_items_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "list_exact_n_items", "item_type": "animals", "count": 2}
    assert rows[0]["prompt"][0]["content"] != "List exactly two animals."
    assert rows[0]["id"] == "dpo_list_exact_n_items_000001"


def test_build_list_exact_n_items_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "list_exact_n_items": [
                {
                    "id": "format_three_colors",
                    "prompt": "List exactly three colors.",
                    "answer": None,
                    "holdout_key": {"type": "list_exact_n_items", "item_type": "colors", "count": 3},
                }
            ]
        }
    )

    rows = build_list_exact_n_items_rows(count=1, holdout_registry=registry)

    assert "colors" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["chosen"][0]["content"] == "dog, cat"
    assert rows[0]["chosen"] != rows[0]["rejected"]


def test_build_private_or_unverifiable_company_fact_rows_creates_valid_dpo_rows():
    rows = build_private_or_unverifiable_company_fact_rows(count=2)

    assert [row["id"] for row in rows] == [
        "dpo_private_or_unverifiable_company_fact_000001",
        "dpo_private_or_unverifiable_company_fact_000002",
    ]
    assert rows[0]["prompt"][0]["content"] == "What was Cohere's private revenue last week?"
    assert rows[0]["chosen"][0]["content"] == "I do not have access to that private or non-public information."
    assert rows[0]["rejected"][0]["content"] == "Cohere's private revenue last week was $18 million."
    assert rows[0]["metadata"]["category"] == "private_info_restraint"
    assert rows[0]["metadata"]["failure_mode"] == "unknown_fact_fabrication"
    assert rows[0]["metadata"]["template_family"] == "private_company_fact"
    assert rows[0]["metadata"]["eval_family"] == "private_or_unverifiable_company_fact"


def test_build_private_or_unverifiable_company_fact_rows_skips_holdout_keys():
    registry = RejectFirstPrivateCompanyRegistry()

    rows = build_private_or_unverifiable_company_fact_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {
        "type": "unverifiable_company_fact",
        "company": "Cohere",
        "fact": "private_revenue",
        "time_window": "last_week",
    }
    assert rows[0]["prompt"][0]["content"] != "What was Cohere's private revenue last week?"
    assert rows[0]["id"] == "dpo_private_or_unverifiable_company_fact_000001"


def test_build_private_or_unverifiable_company_fact_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "private_or_unverifiable_company_fact": [
                {
                    "id": "factual_restraint_private_revenue",
                    "prompt": "What was Anthropic's private revenue last month?",
                    "answer": None,
                    "holdout_key": {
                        "type": "unverifiable_company_fact",
                        "company": "Anthropic",
                        "fact": "private_revenue",
                        "time_window": "last_month",
                    },
                }
            ]
        }
    )

    rows = build_private_or_unverifiable_company_fact_rows(count=1, holdout_registry=registry)

    assert "Anthropic" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["chosen"][0]["content"] == "I do not have access to that private or non-public information."
    assert rows[0]["chosen"] != rows[0]["rejected"]
