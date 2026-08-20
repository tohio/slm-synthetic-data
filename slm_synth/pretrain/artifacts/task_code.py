from __future__ import annotations

from slm_synth.pretrain.artifacts.base import GroundedArtifact
from slm_synth.pretrain.artifacts.task_code_catalog import TASK_CODE_SPECS


class TaskCodeArtifactFactory:
    """Create a finite catalog of distinct, valid Python algorithm records."""

    SPECS = TASK_CODE_SPECS
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
