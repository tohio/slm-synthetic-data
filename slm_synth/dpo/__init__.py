"""Generic DPO dataset helpers."""

from slm_synth.dpo.acceptance import build_dpo_content_summary, partition_unique_dpo_rows
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.dpo.schema import validate_dpo_row, validate_message_list

__all__ = [
    "build_coverage_report",
    "build_dpo_content_summary",
    "partition_unique_dpo_rows",
    "read_jsonl",
    "validate_dpo_row",
    "validate_message_list",
    "write_coverage_report",
    "write_jsonl",
]
