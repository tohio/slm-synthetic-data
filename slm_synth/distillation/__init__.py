"""Response-distillation dataset helpers.

This package is intentionally separate from pretraining synthetic generation.
It contains schema and merge/validation utilities for response-oriented
teacher outputs.
"""

from slm_synth.distillation.schema import (
    FORBIDDEN_PUBLIC_ROW_FIELDS,
    validate_public_row,
)
from slm_synth.distillation.signals import DISTILLATION_SIGNALS, validate_signal
from slm_synth.distillation.io import write_jsonl, write_manifest, write_signal_dataset
from slm_synth.distillation.validate import merge_teacher_outputs

__all__ = [
    "DISTILLATION_SIGNALS",
    "FORBIDDEN_PUBLIC_ROW_FIELDS",
    "merge_teacher_outputs",
    "write_jsonl",
    "write_manifest",
    "write_signal_dataset",
    "validate_public_row",
    "validate_signal",
]
