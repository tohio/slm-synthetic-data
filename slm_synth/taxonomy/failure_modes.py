"""DPO rejected-answer failure mode labels."""

FAILURE_MODES = frozenset(
    {
        "extra_explanation",
        "wrong_numeric_answer",
        "format_violation",
        "unsafe_private_info_guess",
        "unknown_fact_fabrication",
        "future_event_fabrication",
        "persona_fabrication",
        "over_refusal",
        "under_refusal",
        "incomplete_instruction_guess",
        "verbosity_mismatch",
        "code_syntax_error",
        "code_logic_error",
        "code_includes_explanation",
        "wrong_factual_answer",
    }
)


def validate_failure_mode(failure_mode: str) -> str:
    """Return a normalized failure-mode label or raise for unsupported labels."""
    if not isinstance(failure_mode, str):
        raise TypeError("failure_mode must be a string")

    normalized = failure_mode.strip().lower()
    if normalized not in FAILURE_MODES:
        supported = ", ".join(sorted(FAILURE_MODES))
        raise ValueError(f"Unsupported failure_mode '{failure_mode}'. Supported failure modes: {supported}")
    return normalized
