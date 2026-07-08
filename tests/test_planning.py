import pytest

from slm_synth.planning import build_count_plan


def test_build_count_plan_splits_target_deterministically():
    plan = build_count_plan(
        keys=("alpha", "beta", "gamma"),
        target_count=8,
        key_name="family",
        target_count_name="target_rows",
        target_mode="target_rows",
    )

    assert plan.planning_mode == "target_rows"
    assert plan.target_count == 8
    assert plan.count_per_key is None
    assert plan.planned_count == 8
    assert plan.counts_by_key == {"alpha": 3, "beta": 3, "gamma": 2}


def test_build_count_plan_supports_count_per_key():
    plan = build_count_plan(
        keys=("alpha", "beta"),
        count_per_key=2,
        count_per_key_name="count_per_family",
    )

    assert plan.planning_mode == "count_per_family"
    assert plan.target_count is None
    assert plan.count_per_key == 2
    assert plan.planned_count == 4
    assert plan.counts_by_key == {"alpha": 2, "beta": 2}


def test_build_count_plan_requires_exactly_one_strategy():
    with pytest.raises(ValueError, match="provide exactly one"):
        build_count_plan(keys=("alpha",), count_per_key=1, target_count=1)

    with pytest.raises(ValueError, match="provide exactly one"):
        build_count_plan(keys=("alpha",))


def test_build_count_plan_requires_target_for_each_key():
    with pytest.raises(ValueError, match="target_rows must be at least"):
        build_count_plan(
            keys=("alpha", "beta"),
            target_count=1,
            key_name="family",
            target_count_name="target_rows",
        )
