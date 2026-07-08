"""Supported response-distillation signal names."""

DISTILLATION_SIGNALS = frozenset(
    {
        "arithmetic",
        "code",
        "debugging",
        "database",
        "cloud",
        "data_transform",
        "educational_qa",
        "factual_restraint",
        "planning",
        "instruction",
    }
)


def validate_signal(signal: str) -> str:
    """Return a normalized signal name or raise for unsupported signals."""
    if not isinstance(signal, str):
        raise TypeError("signal must be a string")

    normalized = signal.strip().lower()
    if normalized not in DISTILLATION_SIGNALS:
        supported = ", ".join(sorted(DISTILLATION_SIGNALS))
        raise ValueError(f"Unsupported distillation signal '{signal}'. Supported signals: {supported}")
    return normalized
