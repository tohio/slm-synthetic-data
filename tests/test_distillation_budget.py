import pytest

from slm_synth.distillation.budget import (
    TOKEN_TARGETS,
    build_token_budget_plan,
    format_token_count,
    normalize_token_target,
    parse_token_count,
)


def test_named_token_targets_match_distillation_plan():
    assert TOKEN_TARGETS == {
        "smoke": 100_000,
        "pilot": 1_000_000,
        "scale-check": 10_000_000,
        "final": 100_000_000,
    }


def test_parse_token_count_accepts_compact_labels():
    assert parse_token_count("100K") == 100_000
    assert parse_token_count("1M") == 1_000_000
    assert parse_token_count("10M") == 10_000_000
    assert parse_token_count("100000") == 100_000


def test_normalize_token_target_accepts_named_and_numeric_targets():
    assert normalize_token_target("smoke") == ("smoke", 100_000)
    assert normalize_token_target("scale_check") == ("scale-check", 10_000_000)
    assert normalize_token_target("250K") == ("250K", 250_000)
    assert normalize_token_target(1234) == ("1234", 1234)


def test_format_token_count_uses_compact_common_units():
    assert format_token_count(100_000) == "100K"
    assert format_token_count(1_000_000) == "1M"
    assert format_token_count(10_000_000) == "10M"
    assert format_token_count(1234) == "1234"


def test_build_token_budget_plan_distributes_rows_across_requested_signals():
    plan = build_token_budget_plan(
        target="100K",
        signals=["cloud", "database", "debugging"],
        estimated_tokens_per_row=10_000,
    )

    assert plan.target_name == "100K"
    assert plan.total_tokens == 100_000
    assert plan.total_rows == 10
    assert plan.estimated_total_tokens == 100_000
    assert plan.counts_by_signal == {"cloud": 4, "database": 3, "debugging": 3}


def test_build_token_budget_plan_keeps_at_least_one_row_per_signal():
    plan = build_token_budget_plan(
        target=2,
        signals=["cloud", "database", "debugging"],
        estimated_tokens_per_row=10_000,
    )

    assert plan.total_rows == 3
    assert plan.counts_by_signal == {"cloud": 1, "database": 1, "debugging": 1}


def test_build_token_budget_plan_rejects_invalid_estimate():
    with pytest.raises(ValueError, match="estimated_tokens_per_row"):
        build_token_budget_plan(target="smoke", estimated_tokens_per_row=0)
