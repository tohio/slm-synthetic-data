"""Eval-shaped behavior coverage labels."""

EVAL_FAMILIES = frozenset(
    {
        "capital_city_qa",
        "basic_arithmetic_qa",
        "clear_sky_color_qa",
        "ai_concept_explanation",
        "private_or_unverifiable_company_fact",
        "code_generation_function",
        "function_completion_body_only",
        "code_explanation_no_code",
        "code_expression_result",
        "short_factual_stop_behavior",
        "direct_subtraction",
        "direct_division",
        "repeat_exact_n_times",
        "list_exact_n_items",
    }
)


def validate_eval_family(eval_family: str | None) -> str | None:
    """Return a normalized eval-family label, or None when unset."""
    if eval_family is None:
        return None
    if not isinstance(eval_family, str):
        raise TypeError("eval_family must be a string or null")

    normalized = eval_family.strip().lower()
    if not normalized:
        return None
    if normalized not in EVAL_FAMILIES:
        supported = ", ".join(sorted(EVAL_FAMILIES))
        raise ValueError(f"Unsupported eval_family '{eval_family}'. Supported eval families: {supported}")
    return normalized
