from slm_synth.distillation_sft.response_quality import (
    aggregate_rejection_reasons,
    filter_public_rows_by_response_quality,
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
