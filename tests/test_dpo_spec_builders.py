import json

import pytest

from slm_synth.dpo.generation import read_specs_jsonl
from slm_synth.dpo.spec_builders import (
    DPO_SPEC_FAMILIES,
    build_and_write_specs,
    build_specs,
    write_specs_jsonl,
)
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec


@pytest.mark.parametrize("family", sorted(DPO_SPEC_FAMILIES))
def test_build_dpo_specs_for_each_family(family):
    specs = build_specs(family=family, count=2, start_index=3)

    assert len(specs) == 2
    assert specs[0]["id"] == f"dpo_{family}_000003"
    assert specs[1]["id"] == f"dpo_{family}_000004"
    for spec in specs:
        validated = validate_dpo_spec(spec)
        assert validated["metadata"]["eval_family"] == family
        assert validated["metadata"]["failure_mode"]
        assert "preferred assistant response" in validated["instruction"]


def test_dpo_spec_builder_keeps_holdout_key_local_only():
    spec = build_specs(family="basic_arithmetic_qa", count=1)[0]
    visible = teacher_visible_dpo_spec(spec)

    assert "holdout_key" in spec
    assert "holdout_key" not in visible
    assert visible["metadata"]["failure_mode"] == "wrong_numeric_answer"


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("basic_arithmetic_qa", "23"),
        ("direct_subtraction", "28"),
        ("direct_division", "5"),
        ("code_expression_result", "15"),
        ("repeat_exact_n_times", "dog dog dog dog"),
        ("list_exact_n_items", "red, green, blue, purple"),
    ],
)
def test_dpo_spec_builder_adds_rejected_answer_for_exact_families(family, expected):
    spec = build_specs(family=family, count=1)[0]

    assert spec["variables"]["rejected_answer"] == expected
    assert "Rejected assistant content must exactly match variables.rejected_answer." in spec["constraints"]


def test_write_dpo_specs_jsonl_round_trips_through_generation_reader(tmp_path):
    specs = build_specs(family="repeat_exact_n_times", count=2)
    path = tmp_path / "dpo.specs.jsonl"

    count = write_specs_jsonl(specs, path)

    assert count == 2
    rows = read_specs_jsonl(path)
    assert [row["id"] for row in rows] == [
        "dpo_repeat_exact_n_times_000001",
        "dpo_repeat_exact_n_times_000002",
    ]
    assert rows[0]["metadata"]["failure_mode"] == "format_violation"


def test_build_and_write_dpo_specs_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dpo.specs.jsonl"

    count = build_and_write_specs(
        family="code_expression_result",
        count=1,
        output_path=path,
        start_index=8,
    )

    assert count == 1
    row = json.loads(path.read_text().strip())
    assert row["id"] == "dpo_code_expression_result_000008"
    assert row["metadata"]["failure_mode"] == "wrong_numeric_answer"


def test_build_dpo_specs_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported DPO spec family"):
        build_specs(family="unknown_family", count=1)


def test_build_dpo_specs_rejects_bad_count():
    with pytest.raises(ValueError, match="count"):
        build_specs(family="basic_arithmetic_qa", count=0)


def test_function_completion_dpo_specs_ground_signature_and_exact_body():
    spec = build_specs(family="function_completion_body_only", count=1)[0]

    assert spec["variables"]["function_signature"] == "def add_numbers(a, b):"
    assert spec["variables"]["chosen_answer"] == "return a + b"
    assert spec["variables"]["rejected_answer"] == "# Explanation: implement the requested behavior.\nreturn a + b"
    assert any("Prompt must include variables.function_signature" in c for c in spec["constraints"])
    assert any("Chosen response must be semantically correct" in c for c in spec["constraints"])


def test_code_generation_dpo_specs_require_complete_function_request():
    spec = build_specs(family="code_generation_function", count=1)[0]

    assert spec["variables"]["function_signature"] == "def add_numbers(a, b):"
    assert spec["variables"]["chosen_answer"] == "def add_numbers(a, b):\n    return a + b"
    assert spec["variables"]["rejected_answer"].startswith("Here is the function:")
    assert any("Prompt must ask for a complete Python function" in c for c in spec["constraints"])


def test_list_exact_dpo_specs_forbid_prompt_answer_leakage():
    spec = build_specs(family="list_exact_n_items", count=1)[0]

    assert spec["variables"]["chosen_answer"] == "red, green, blue"
    assert spec["variables"]["rejected_answer"] == "red, green, blue, purple"
    assert any("must not reveal variables.items or variables.answer" in c for c in spec["constraints"])


def test_short_factual_dpo_rejected_answer_matches_verbosity_failure_mode():
    spec = build_specs(family="short_factual_stop_behavior", count=1)[0]

    assert spec["variables"]["chosen_answer"] == spec["variables"]["capital"]
    assert spec["variables"]["rejected_answer"] == "The capital of Italy is Rome."
