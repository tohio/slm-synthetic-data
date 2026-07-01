"""Training-objective category labels."""

CATEGORIES = frozenset(
    {
        "general_instruction_following",
        "concise_factual_qa",
        "atomic_factual_mapping",
        "cloze_completion",
        "answer_only_compliance",
        "direct_arithmetic",
        "word_problem_arithmetic",
        "decimal_comparison",
        "exact_output_format_control",
        "code_generation",
        "code_expression_evaluation",
        "private_info_restraint",
        "unknown_fact_restraint",
        "future_event_restraint",
        "incomplete_prompt_handling",
        "no_persona_fabrication",
        "refusal_calibration",
        "controlled_verbosity",
    }
)


def validate_category(category: str) -> str:
    """Return a normalized category label or raise for unsupported labels."""
    if not isinstance(category, str):
        raise TypeError("category must be a string")

    normalized = category.strip().lower()
    if normalized not in CATEGORIES:
        supported = ", ".join(sorted(CATEGORIES))
        raise ValueError(f"Unsupported category '{category}'. Supported categories: {supported}")
    return normalized
