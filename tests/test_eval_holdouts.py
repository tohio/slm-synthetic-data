import pytest

from slm_synth.taxonomy.holdouts import (
    HoldoutRegistry,
    holdout_key_fingerprint,
    load_default_holdout_registry,
    normalize_text,
)


def test_normalize_text_collapses_case_and_whitespace():
    assert normalize_text("  What is 2 + 2?\n") == "what is 2 + 2?"


def test_holdout_key_fingerprint_is_order_stable():
    left = {"type": "arithmetic", "operation": "add", "operands": [2, 2]}
    right = {"operands": [2, 2], "operation": "add", "type": "arithmetic"}

    assert holdout_key_fingerprint(left) == holdout_key_fingerprint(right)


def test_default_registry_loads_sanity_eval_holdouts():
    registry = load_default_holdout_registry()

    assert registry.contains_prompt("What is 2 + 2?")
    assert registry.contains_prompt("Repeat cat exactly three times.")
    assert registry.contains_prompt("List exactly three colors.")
    assert registry.contains_prompt("What was Anthropic's private revenue last month?")


def test_registry_rejects_exact_prompt_but_allows_sibling_prompt():
    registry = load_default_holdout_registry()

    with pytest.raises(ValueError, match="prompt matches"):
        registry.reject_if_holdout(prompt="  what is 2 + 2? ")

    registry.reject_if_holdout(prompt="What is 6 + 9?")


def test_registry_rejects_structured_holdout_key_but_allows_sibling_key():
    registry = load_default_holdout_registry()

    with pytest.raises(ValueError, match="holdout_key matches"):
        registry.reject_if_holdout(
            prompt="Can you calculate 2 + 2?",
            holdout_key={"type": "arithmetic", "operation": "add", "operands": [2, 2]},
        )

    registry.reject_if_holdout(
        prompt="Can you calculate 6 + 9?",
        holdout_key={"type": "arithmetic", "operation": "add", "operands": [6, 9]},
    )


def test_registry_can_load_minimal_mapping():
    registry = HoldoutRegistry.from_mapping(
        {
            "repeat_exact_n_times": [
                {
                    "id": "repeat_cat",
                    "prompt": "Repeat cat exactly three times.",
                    "answer": "cat cat cat",
                    "holdout_key": {"type": "repeat_exact_n_times", "token": "cat", "count": 3},
                }
            ]
        }
    )

    assert registry.contains_prompt("repeat cat exactly three times.")
    assert registry.contains_holdout_key({"count": 3, "token": "cat", "type": "repeat_exact_n_times"})
