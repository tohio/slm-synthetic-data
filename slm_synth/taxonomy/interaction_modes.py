"""Cross-cutting conversational interaction modes."""

from collections.abc import Sequence

INTERACTION_MODES = frozenset({"single_turn", "multi_turn", "system_conditioned", "tool_mediated"})


def validate_interaction_modes(value: Sequence[str]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError("interaction_modes must be a non-empty list")
    modes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("interaction_modes entries must be strings")
        normalized = item.strip().lower()
        if normalized not in INTERACTION_MODES:
            raise ValueError(f"Unsupported interaction mode {item!r}")
        if normalized not in modes:
            modes.append(normalized)
    if "single_turn" in modes and "multi_turn" in modes:
        raise ValueError("interaction_modes cannot contain both single_turn and multi_turn")
    if not ({"single_turn", "multi_turn"} & set(modes)):
        raise ValueError("interaction_modes must contain single_turn or multi_turn")
    return modes
