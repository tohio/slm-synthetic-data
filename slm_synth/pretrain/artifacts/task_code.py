from __future__ import annotations

import hashlib

from slm_synth.pretrain.artifacts.base import GroundedArtifact
from slm_synth.pretrain.artifacts.task_code_catalog import TASK_CODE_SPECS


def _catalog_order(spec: tuple[str, str, str]) -> bytes:
    """Return a stable order that samples across the whole catalog, not its source layout."""
    return hashlib.sha256(f"task-code-candidate:{spec[0]}".encode("utf-8")).digest()


class TaskCodeArtifactFactory:
    """Create a finite catalog of distinct, valid Python algorithm records."""

    SPECS = tuple(sorted(TASK_CODE_SPECS, key=_catalog_order))
    FAMILIES = tuple(spec[0] for spec in SPECS)
    UNIQUE_CANDIDATE_CAPACITY = len(SPECS)

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"task_code index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        family, task, code = self.SPECS[index]
        return GroundedArtifact(
            signal="task_code",
            family=family,
            artifact_id=f"task_code_{family}_{index + 1:09d}",
            payload={"task": task, "code": code},
        )
