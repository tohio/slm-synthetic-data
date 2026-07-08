"""Response-distillation dataset helpers.

This package is intentionally separate from pretraining synthetic generation.
It contains schema and merge/validation utilities for response-oriented
teacher outputs.
"""

from slm_synth.distillation_sft.schema import (
    FORBIDDEN_PUBLIC_ROW_FIELDS,
    validate_public_row,
)
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal
from slm_synth.distillation_sft.io import write_jsonl, write_manifest, write_signal_dataset
from slm_synth.distillation_sft.report import build_coverage_report, write_coverage_report
from slm_synth.distillation_sft.validate import merge_teacher_outputs

__all__ = [
    "DISTILLATION_SIGNALS",
    "FORBIDDEN_PUBLIC_ROW_FIELDS",
    "build_coverage_report",
    "merge_teacher_outputs",
    "write_jsonl",
    "write_manifest",
    "write_coverage_report",
    "write_signal_dataset",
    "validate_public_row",
    "validate_signal",
]
