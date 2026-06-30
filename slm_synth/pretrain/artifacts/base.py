from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundedArtifact:
    """A deterministic fact-bearing input rendered into a natural final record."""

    signal: str
    family: str
    artifact_id: str
    payload: dict[str, Any]
