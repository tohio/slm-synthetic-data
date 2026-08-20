"""Cross-cutting assistant output modes."""

OUTPUT_MODES = frozenset({"free_text", "concise", "structured_json", "table", "exact_constraints", "code", "tool_call"})


def validate_output_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_mode must be a string")
    normalized = value.strip().lower()
    if normalized not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output_mode {value!r}")
    return normalized
