"""Broad, non-overlapping task families for generic SFT and DPO data."""

TASK_FAMILIES = frozenset(
    {
        "everyday_conversation",
        "rewriting_and_editing",
        "summarization",
        "classification_and_extraction",
        "grounded_qa_and_reading",
        "planning_brainstorming_recommendations",
        "creative_writing",
        "programming",
        "applied_math_and_reasoning",
        "safety_uncertainty_and_refusal",
    }
)


def validate_task_family(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("task_family must be a string")
    normalized = value.strip().lower()
    if normalized not in TASK_FAMILIES:
        raise ValueError(f"Unsupported task_family {value!r}. Supported task families: {', '.join(sorted(TASK_FAMILIES))}")
    return normalized
