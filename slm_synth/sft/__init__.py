"""Generic SFT dataset generation, validation, reporting, and publication."""

from slm_synth.sft.acceptance import (
    build_sft_content_summary,
    normalize_sft_text,
    partition_unique_sft_rows,
    sft_conversation_fingerprint,
    sft_prompt_fingerprint,
    sft_response_fingerprint,
)
from slm_synth.sft.card import load_sft_dataset_card_configs, require_sft_dataset_card_configs
from slm_synth.sft.report import build_coverage_report, require_publish_ready_report, write_coverage_report
from slm_synth.sft.schema import validate_message, validate_sft_row
from slm_synth.sft.spec_builders import SFT_SPEC_FAMILIES, build_specs
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec

__all__ = [
    "SFT_SPEC_FAMILIES",
    "build_coverage_report",
    "build_sft_content_summary",
    "build_specs",
    "load_sft_dataset_card_configs",
    "normalize_sft_text",
    "partition_unique_sft_rows",
    "require_publish_ready_report",
    "require_sft_dataset_card_configs",
    "sft_conversation_fingerprint",
    "sft_prompt_fingerprint",
    "sft_response_fingerprint",
    "teacher_visible_sft_spec",
    "validate_message",
    "validate_sft_row",
    "validate_sft_spec",
    "write_coverage_report",
]
