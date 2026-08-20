"""Cross-cutting source-context modes."""

CONTEXT_MODES = frozenset({"self_contained", "supplied_passage", "long_document", "multi_document"})


def validate_context_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("context_mode must be a string")
    normalized = value.strip().lower()
    if normalized not in CONTEXT_MODES:
        raise ValueError(f"Unsupported context_mode {value!r}")
    return normalized
