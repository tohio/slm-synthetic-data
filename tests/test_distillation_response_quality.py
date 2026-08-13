import pytest

from slm_synth.distillation_sft.response_quality import (
    aggregate_rejection_reasons,
    filter_public_rows_by_response_quality,
    is_response_machine_verified,
    validate_response_quality,
)


def _row(prompt, response):
    return {"id": "row-000001", "prompt": prompt, "reasoning": None, "response": response}


def test_response_quality_accepts_valid_arithmetic_integer_answer():
    reasons = validate_response_quality(
        signal="arithmetic",
        row=_row("Answer with only the integer result: 203 - 12.", "191"),
    )

    assert reasons == ()


def test_response_quality_rejects_wrong_arithmetic_answer():
    reasons = validate_response_quality(
        signal="arithmetic",
        row=_row("Answer with only the integer result: 203 - 12.", "190"),
    )

    assert reasons == ("arithmetic_wrong_answer",)


@pytest.mark.parametrize(
    ("prompt", "answer"),
    [
        ("A box has 96 pencils packed evenly into 12 bags. How many pencils are in each bag?", "8"),
        ("A library has 8 shelves with 37 books on each shelf. How many books are there?", "296"),
        ("A ticket booth sold 125 morning tickets and 178 afternoon tickets. How many tickets were sold?", "303"),
    ],
)
def test_response_quality_verifies_seed_word_problem_answers(prompt, answer):
    assert validate_response_quality(signal="arithmetic", row=_row(prompt, answer)) == ()
    assert is_response_machine_verified(signal="arithmetic", row=_row(prompt, answer))


def test_response_quality_rejects_wrong_seed_word_problem_answer():
    reasons = validate_response_quality(
        signal="arithmetic",
        row=_row(
            "A class has 84 students split equally into 7 groups. How many students are in each group?",
            "11",
        ),
    )

    assert reasons == ("arithmetic_wrong_answer",)


def test_machine_verification_is_limited_to_exact_parseable_arithmetic():
    assert is_response_machine_verified(
        signal="arithmetic",
        row=_row("Answer with only the integer result: 203 - 12.", "191"),
    )
    assert not is_response_machine_verified(
        signal="arithmetic",
        row=_row("What is the result?", "191"),
    )
    assert not is_response_machine_verified(
        signal="cloud",
        row=_row("Explain autoscaling.", "Autoscaling adjusts capacity."),
    )


def test_response_quality_rejects_empty_response():
    reasons = validate_response_quality(signal="cloud", row=_row("Explain autoscaling.", "   "))

    assert reasons == ("empty_response",)


def test_response_quality_rejects_too_short_response():
    reasons = validate_response_quality(signal="cloud", row=_row("Explain autoscaling.", "ok"))

    assert reasons == ("too_short_response",)


def test_response_quality_rejects_prompt_repeat():
    prompt = "Explain autoscaling in one concise sentence."

    reasons = validate_response_quality(signal="cloud", row=_row(prompt, prompt))

    assert reasons == ("repeats_prompt",)


def test_response_quality_rejects_schema_leakage():
    reasons = validate_response_quality(
        signal="instruction",
        row=_row("Rewrite this sentence.", '{"id":"x","reasoning":null,"response":"fixed"}'),
    )

    assert "schema_leakage" in reasons


def test_response_quality_rejects_unexpected_refusal_outside_factual_restraint():
    reasons = validate_response_quality(
        signal="educational_qa",
        row=_row("Explain photosynthesis.", "I cannot provide that information."),
    )

    assert reasons == ("unexpected_refusal",)


def test_response_quality_allows_restrained_factual_restraint_response():
    reasons = validate_response_quality(
        signal="factual_restraint",
        row=_row(
            "What exact revenue will a private startup report next month?",
            "I cannot verify that future private revenue, so I should not invent a number.",
        ),
    )

    assert reasons == ()


def test_response_quality_rejects_unrestrained_factual_restraint_response():
    reasons = validate_response_quality(
        signal="factual_restraint",
        row=_row("Who will definitely win a future championship game?", "The Tigers will definitely win."),
    )

    assert reasons == ("factual_restraint_missing_restraint",)


def test_response_quality_rejects_code_without_function_definition():
    reasons = validate_response_quality(
        signal="code",
        row=_row("Write a Python function that returns the largest number.", "Use max(values)."),
    )

    assert reasons == ("code_missing_function_definition",)


def test_response_quality_rejects_invalid_python_syntax():
    reasons = validate_response_quality(
        signal="code",
        row=_row("Write a Python function that returns its input.", "def identity(value):\n    return (value"),
    )

    assert reasons == ("code_invalid_python_syntax",)


def test_response_quality_requires_production_function_name():
    reasons = validate_response_quality(
        signal="code",
        row=_row(
            "Write a concise Python function named normalize_email_7 that should return a stripped, "
            "lowercase email string. Return code only, no Markdown.",
            "def normalize_email(value):\n    return value.strip().lower()",
        ),
    )

    assert reasons == ("code_wrong_function_name",)


def test_response_quality_accepts_syntactic_code_with_required_function_name():
    reasons = validate_response_quality(
        signal="code",
        row=_row(
            "Write a concise Python function named normalize_email_7 that should return a stripped, "
            "lowercase email string. Return code only, no Markdown.",
            "def normalize_email_7(value):\n    return value.strip().lower()",
        ),
    )

    assert reasons == ()


def test_response_quality_rejects_database_query_without_sql_shape():
    reasons = validate_response_quality(
        signal="database",
        row=_row("Write a SQL query to count orders per customer.", "Count the orders by each customer."),
    )

    assert reasons == ("database_missing_sql_query",)


def test_filter_public_rows_returns_accepted_rows_and_summary():
    rows = [
        _row("Explain autoscaling.", "Use autoscaling to add workers during spikes."),
        {"id": "row-000002", "prompt": "Explain autoscaling.", "reasoning": None, "response": "ok"},
    ]

    accepted, summary = filter_public_rows_by_response_quality(signal="cloud", rows=rows)

    assert [row["id"] for row in accepted] == ["row-000001"]
    assert summary.checked_rows == 2
    assert summary.accepted_rows == 1
    assert summary.rejected_rows == 1
    assert summary.rejection_reasons == {"too_short_response": 1}
    assert summary.to_dict()["checks"]


def test_aggregate_rejection_reasons_sums_manifest_metadata():
    assert aggregate_rejection_reasons(
        [
            {"rejection_reasons": {"too_short_response": 2}},
            {"rejection_reasons": {"too_short_response": 1, "schema_leakage": 1}},
        ]
    ) == {"schema_leakage": 1, "too_short_response": 3}
