"""Template-family validation.

Template families are intentionally open-ended. They identify generator surface
patterns and will grow faster than the stable category and eval-family labels.
"""

import re

_TEMPLATE_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_template_family(template_family: str) -> str:
    """Return a normalized template-family label.

    Template labels are not enumerated here, but they must be stable
    snake_case identifiers so coverage reports can group them reliably.
    """
    if not isinstance(template_family, str):
        raise TypeError("template_family must be a string")

    normalized = template_family.strip().lower()
    if not normalized:
        raise ValueError("template_family must be a non-empty string")
    if not _TEMPLATE_FAMILY_RE.fullmatch(normalized):
        raise ValueError("template_family must be a snake_case identifier")
    return normalized
