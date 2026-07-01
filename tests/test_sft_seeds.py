import pytest

from slm_synth.sft.seeds import (
    build_answer_only_arithmetic_rows,
    build_capital_city_qa_rows,
    build_clear_sky_color_qa_rows,
    build_code_explanation_no_code_rows,
    build_code_generation_function_rows,
    build_function_completion_body_only_rows,
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


class RejectFirstCapitalCityRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {"type": "capital_city", "country": "Italy", "answer": "Rome"}:
            raise ValueError("candidate holdout_key matches eval holdout key")


class RejectFirstSkyColorRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {
            "type": "factual_relation",
            "relation": "cloudless_day_sky_color",
            "answer": "blue",
        }:
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


class RejectFirstCodeGenerationRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {
            "type": "code_generation",
            "function_name": "multiply",
            "function_task": "multiply_two_numbers",
        }:
            raise ValueError("candidate holdout_key matches eval holdout key")


class RejectFirstFunctionCompletionRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {"type": "function_completion", "function_name": "double"}:
            raise ValueError("candidate holdout_key matches eval holdout key")


class RejectFirstCodeExplanationRegistry:
    def __init__(self):
        self.seen = []

    def reject_if_holdout(self, *, prompt, holdout_key=None):
        self.seen.append((prompt, holdout_key))
        if holdout_key == {
            "type": "code_explanation",
            "function_name": "double",
            "operation": "double_number",
        }:
            raise ValueError("candidate holdout_key matches eval holdout key")


def test_build_answer_only_arithmetic_rows_creates_valid_sft_rows():
    rows = build_answer_only_arithmetic_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_answer_only_arithmetic_000001",
        "sft_answer_only_arithmetic_000002",
    ]
    assert all(row["messages"][0]["role"] == "user" for row in rows)
    assert all(row["messages"][1]["role"] == "assistant" for row in rows)
    assert all(row["metadata"]["category"] == "answer_only_compliance" for row in rows)
    assert all(row["metadata"]["eval_family"] == "basic_arithmetic_qa" for row in rows)


def test_build_seed_rows_dispatches_supported_family():
    rows = build_seed_rows(family="answer_only_arithmetic", count=1, start_index=7)

    assert rows[0]["id"] == "sft_answer_only_arithmetic_000007"


def test_build_seed_rows_dispatches_capital_city_family():
    rows = build_seed_rows(family="capital_city_qa", count=1, start_index=7)

    assert rows[0]["id"] == "sft_capital_city_qa_000007"


def test_build_seed_rows_dispatches_sky_color_family():
    rows = build_seed_rows(family="clear_sky_color_qa", count=1, start_index=7)

    assert rows[0]["id"] == "sft_clear_sky_color_qa_000007"


def test_build_seed_rows_dispatches_repeat_family():
    rows = build_seed_rows(family="repeat_exact_n_times", count=1, start_index=7)

    assert rows[0]["id"] == "sft_repeat_exact_n_times_000007"


def test_build_seed_rows_dispatches_list_family():
    rows = build_seed_rows(family="list_exact_n_items", count=1, start_index=7)

    assert rows[0]["id"] == "sft_list_exact_n_items_000007"


def test_build_seed_rows_dispatches_private_company_family():
    rows = build_seed_rows(family="private_or_unverifiable_company_fact", count=1, start_index=7)

    assert rows[0]["id"] == "sft_private_or_unverifiable_company_fact_000007"


def test_build_seed_rows_dispatches_code_generation_family():
    rows = build_seed_rows(family="code_generation_function", count=1, start_index=7)

    assert rows[0]["id"] == "sft_code_generation_function_000007"


def test_build_seed_rows_dispatches_function_completion_family():
    rows = build_seed_rows(family="function_completion_body_only", count=1, start_index=7)

    assert rows[0]["id"] == "sft_function_completion_body_only_000007"


def test_build_seed_rows_dispatches_code_explanation_family():
    rows = build_seed_rows(family="code_explanation_no_code", count=1, start_index=7)

    assert rows[0]["id"] == "sft_code_explanation_no_code_000007"


def test_build_seed_rows_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported SFT seed family"):
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
    assert rows[0]["messages"][0]["content"] != "Answer with only the number: What is 3 + 4?"
    assert rows[0]["id"] == "sft_answer_only_arithmetic_000001"


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

    assert "2 + 2" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "7"


def test_build_capital_city_qa_rows_creates_valid_sft_rows():
    rows = build_capital_city_qa_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_capital_city_qa_000001",
        "sft_capital_city_qa_000002",
    ]
    assert rows[0]["messages"][0]["content"] == "What is the capital of Italy?"
    assert rows[0]["messages"][1]["content"] == "Rome"
    assert rows[0]["metadata"]["category"] == "concise_factual_qa"
    assert rows[0]["metadata"]["template_family"] == "capital_city_direct"
    assert rows[0]["metadata"]["eval_family"] == "capital_city_qa"


def test_build_capital_city_qa_rows_skips_holdout_keys():
    registry = RejectFirstCapitalCityRegistry()

    rows = build_capital_city_qa_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "capital_city", "country": "Italy", "answer": "Rome"}
    assert rows[0]["messages"][0]["content"] != "What is the capital of Italy?"
    assert rows[0]["id"] == "sft_capital_city_qa_000001"


def test_build_capital_city_qa_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "capital_city_qa": [
                {
                    "id": "factual_capital_france",
                    "prompt": "What is the capital of France?",
                    "answer": "Paris",
                    "holdout_key": {"type": "capital_city", "country": "France", "answer": "Paris"},
                },
                {
                    "id": "stop_behavior_short_answer",
                    "prompt": "What is the capital of Japan?",
                    "answer": "Tokyo",
                    "holdout_key": {"type": "capital_city", "country": "Japan", "answer": "Tokyo"},
                },
            ]
        }
    )

    rows = build_capital_city_qa_rows(count=1, holdout_registry=registry)

    assert "France" not in rows[0]["messages"][0]["content"]
    assert "Japan" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "Rome"


def test_build_clear_sky_color_qa_rows_creates_valid_sft_rows():
    rows = build_clear_sky_color_qa_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_clear_sky_color_qa_000001",
        "sft_clear_sky_color_qa_000002",
    ]
    assert rows[0]["messages"][0]["content"] == "On a cloudless day, what color does the sky usually appear?"
    assert rows[0]["messages"][1]["content"] == "blue"
    assert rows[0]["metadata"]["category"] == "concise_factual_qa"
    assert rows[0]["metadata"]["template_family"] == "sky_color_direct"
    assert rows[0]["metadata"]["eval_family"] == "clear_sky_color_qa"


def test_build_clear_sky_color_qa_rows_skips_holdout_keys():
    registry = RejectFirstSkyColorRegistry()

    rows = build_clear_sky_color_qa_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {
        "type": "factual_relation",
        "relation": "cloudless_day_sky_color",
        "answer": "blue",
    }
    assert rows[0]["messages"][0]["content"] != "On a cloudless day, what color does the sky usually appear?"
    assert rows[0]["id"] == "sft_clear_sky_color_qa_000001"


def test_build_clear_sky_color_qa_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "clear_sky_color_qa": [
                {
                    "id": "factual_sky",
                    "prompt": "What color is the sky on a clear day?",
                    "answer": "blue",
                    "holdout_key": {
                        "type": "factual_relation",
                        "relation": "clear_day_sky_color",
                        "answer": "blue",
                    },
                }
            ]
        }
    )

    rows = build_clear_sky_color_qa_rows(count=1, holdout_registry=registry)

    assert rows[0]["messages"][0]["content"] != "What color is the sky on a clear day?"
    assert rows[0]["messages"][1]["content"] == "blue"


def test_build_repeat_exact_n_times_rows_creates_valid_sft_rows():
    rows = build_repeat_exact_n_times_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_repeat_exact_n_times_000001",
        "sft_repeat_exact_n_times_000002",
    ]
    assert rows[0]["messages"][0]["content"] == "Repeat dog exactly two times."
    assert rows[0]["messages"][1]["content"] == "dog dog"
    assert rows[0]["metadata"]["category"] == "exact_output_format_control"
    assert rows[0]["metadata"]["template_family"] == "repeat_word_count"
    assert rows[0]["metadata"]["eval_family"] == "repeat_exact_n_times"


def test_build_repeat_exact_n_times_rows_skips_holdout_keys():
    registry = RejectFirstRepeatRegistry()

    rows = build_repeat_exact_n_times_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "repeat_exact_n_times", "token": "dog", "count": 2}
    assert rows[0]["messages"][0]["content"] != "Repeat dog exactly two times."
    assert rows[0]["id"] == "sft_repeat_exact_n_times_000001"


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

    assert "cat" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "dog dog"


def test_build_list_exact_n_items_rows_creates_valid_sft_rows():
    rows = build_list_exact_n_items_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_list_exact_n_items_000001",
        "sft_list_exact_n_items_000002",
    ]
    assert rows[0]["messages"][0]["content"] == "List exactly two animals."
    assert rows[0]["messages"][1]["content"] == "dog, cat"
    assert rows[0]["metadata"]["category"] == "exact_output_format_control"
    assert rows[0]["metadata"]["template_family"] == "list_n_items"
    assert rows[0]["metadata"]["eval_family"] == "list_exact_n_items"


def test_build_list_exact_n_items_rows_skips_holdout_keys():
    registry = RejectFirstListRegistry()

    rows = build_list_exact_n_items_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "list_exact_n_items", "item_type": "animals", "count": 2}
    assert rows[0]["messages"][0]["content"] != "List exactly two animals."
    assert rows[0]["id"] == "sft_list_exact_n_items_000001"


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

    assert "colors" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "dog, cat"


def test_build_private_or_unverifiable_company_fact_rows_creates_valid_sft_rows():
    rows = build_private_or_unverifiable_company_fact_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_private_or_unverifiable_company_fact_000001",
        "sft_private_or_unverifiable_company_fact_000002",
    ]
    assert rows[0]["messages"][0]["content"] == "What was Cohere's private revenue last week?"
    assert rows[0]["messages"][1]["content"] == "I do not have access to that private or non-public information."
    assert rows[0]["metadata"]["category"] == "private_info_restraint"
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
    assert rows[0]["messages"][0]["content"] != "What was Cohere's private revenue last week?"
    assert rows[0]["id"] == "sft_private_or_unverifiable_company_fact_000001"


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

    assert "Anthropic" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "I do not have access to that private or non-public information."


def test_build_code_generation_function_rows_creates_valid_sft_rows():
    rows = build_code_generation_function_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_code_generation_function_000001",
        "sft_code_generation_function_000002",
    ]
    assert rows[0]["messages"][0]["content"] == (
        "Write a Python function named multiply that multiplies two numbers. Return only code."
    )
    assert rows[0]["messages"][1]["content"] == "def multiply(a, b):\n    return a * b"
    assert rows[0]["metadata"]["category"] == "code_generation"
    assert rows[0]["metadata"]["template_family"] == "simple_function_generation"
    assert rows[0]["metadata"]["eval_family"] == "code_generation_function"


def test_build_code_generation_function_rows_skips_holdout_keys():
    registry = RejectFirstCodeGenerationRegistry()

    rows = build_code_generation_function_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {
        "type": "code_generation",
        "function_name": "multiply",
        "function_task": "multiply_two_numbers",
    }
    assert rows[0]["messages"][0]["content"] != (
        "Write a Python function named multiply that multiplies two numbers. Return only code."
    )
    assert rows[0]["id"] == "sft_code_generation_function_000001"


def test_build_code_generation_function_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "code_generation_function": [
                {
                    "id": "code_generation_square",
                    "prompt": "Write a Python function named square that returns the square of a number. Return only code.",
                    "answer": "def square",
                    "holdout_key": {
                        "type": "code_generation",
                        "function_name": "square",
                        "function_task": "square_number",
                    },
                }
            ]
        }
    )

    rows = build_code_generation_function_rows(count=1, holdout_registry=registry)

    assert "square" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"].startswith("def multiply")


def test_build_function_completion_body_only_rows_creates_valid_sft_rows():
    rows = build_function_completion_body_only_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_function_completion_body_only_000001",
        "sft_function_completion_body_only_000002",
    ]
    assert "def double(x):" in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "return x * 2"
    assert rows[0]["metadata"]["category"] == "code_generation"
    assert rows[0]["metadata"]["template_family"] == "function_body_completion"
    assert rows[0]["metadata"]["eval_family"] == "function_completion_body_only"


def test_build_function_completion_body_only_rows_skips_holdout_keys():
    registry = RejectFirstFunctionCompletionRegistry()

    rows = build_function_completion_body_only_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {"type": "function_completion", "function_name": "double"}
    assert "def double(x):" not in rows[0]["messages"][0]["content"]
    assert rows[0]["id"] == "sft_function_completion_body_only_000001"


def test_build_function_completion_body_only_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "function_completion_body_only": [
                {
                    "id": "function_completion_body",
                    "prompt": "Complete this Python function. Return only the function body.\n\ndef first_item(items):",
                    "answer": "return",
                    "holdout_key": {"type": "function_completion", "function_name": "first_item"},
                },
                {
                    "id": "function_completion_has_close",
                    "prompt": "Complete this Python function. Return only the function body.\n\ndef has_close_elements(numbers, threshold):",
                    "answer": "return",
                    "holdout_key": {"type": "function_completion", "function_name": "has_close_elements"},
                },
            ]
        }
    )

    rows = build_function_completion_body_only_rows(count=1, holdout_registry=registry)

    assert "first_item" not in rows[0]["messages"][0]["content"]
    assert "has_close_elements" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "return x * 2"


def test_build_code_explanation_no_code_rows_creates_valid_sft_rows():
    rows = build_code_explanation_no_code_rows(count=2)

    assert [row["id"] for row in rows] == [
        "sft_code_explanation_no_code_000001",
        "sft_code_explanation_no_code_000002",
    ]
    assert "def double" in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "It returns the input value multiplied by two."
    assert "def double" not in rows[0]["messages"][1]["content"]
    assert "```" not in rows[0]["messages"][1]["content"]
    assert rows[0]["metadata"]["category"] == "general_instruction_following"
    assert rows[0]["metadata"]["template_family"] == "plain_code_explanation"
    assert rows[0]["metadata"]["eval_family"] == "code_explanation_no_code"


def test_build_code_explanation_no_code_rows_skips_holdout_keys():
    registry = RejectFirstCodeExplanationRegistry()

    rows = build_code_explanation_no_code_rows(
        count=1,
        holdout_registry=registry,  # type: ignore[arg-type]
    )

    assert registry.seen[0][1] == {
        "type": "code_explanation",
        "function_name": "double",
        "operation": "double_number",
    }
    assert "def double" not in rows[0]["messages"][0]["content"]
    assert rows[0]["id"] == "sft_code_explanation_no_code_000001"


def test_build_code_explanation_no_code_rows_allows_sibling_not_eval_holdout():
    registry = HoldoutRegistry.from_mapping(
        {
            "code_explanation_no_code": [
                {
                    "id": "code_explanation_square",
                    "prompt": "Explain what this Python function does:\n\ndef square(x):\n    return x * x",
                    "answer": "square",
                    "holdout_key": {
                        "type": "code_explanation",
                        "function_name": "square",
                        "operation": "square_number",
                    },
                }
            ]
        }
    )

    rows = build_code_explanation_no_code_rows(count=1, holdout_registry=registry)

    assert "square" not in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "It returns the input value multiplied by two."
