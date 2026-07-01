"""Difficulty validation for synthetic alignment metadata."""

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5


def validate_difficulty(difficulty: int) -> int:
    """Return a validated difficulty value in the inclusive range 1-5."""
    if not isinstance(difficulty, int) or isinstance(difficulty, bool):
        raise TypeError("difficulty must be an integer")
    if difficulty < MIN_DIFFICULTY or difficulty > MAX_DIFFICULTY:
        raise ValueError(f"difficulty must be between {MIN_DIFFICULTY} and {MAX_DIFFICULTY}")
    return difficulty
