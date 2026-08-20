"""Quality dimensions represented by generic DPO preference pairs."""

PREFERENCE_DIMENSIONS = frozenset(
    {
        "helpfulness_and_completeness",
        "factual_accuracy",
        "instruction_adherence",
        "appropriate_detail",
        "organization",
        "style_and_tone",
        "tool_call_correctness",
        "groundedness",
        "safe_refusal_calibration",
        "code_correctness",
    }
)


def validate_preference_dimension(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("preference_dimension must be a string")
    normalized = value.strip().lower()
    if normalized not in PREFERENCE_DIMENSIONS:
        raise ValueError(f"Unsupported preference_dimension {value!r}")
    return normalized
