"""SFT synthetic row validation."""

from slm_synth.sft.io import read_jsonl, write_jsonl
from slm_synth.sft.manifest import build_manifest_payload, write_manifest
from slm_synth.sft.runs import SFTSeedRunResult, materialize_seed_dataset
from slm_synth.sft.schema import validate_message, validate_sft_row
from slm_synth.sft.seeds import (
    SFT_SEED_FAMILIES,
    build_answer_only_arithmetic_rows,
    build_code_explanation_no_code_rows,
    build_code_generation_function_rows,
    build_function_completion_body_only_rows,
    build_list_exact_n_items_rows,
    build_private_or_unverifiable_company_fact_rows,
    build_repeat_exact_n_times_rows,
    build_seed_rows,
)

__all__ = [
    "SFT_SEED_FAMILIES",
    "SFTSeedRunResult",
    "build_manifest_payload",
    "build_answer_only_arithmetic_rows",
    "build_code_explanation_no_code_rows",
    "build_code_generation_function_rows",
    "build_function_completion_body_only_rows",
    "build_list_exact_n_items_rows",
    "build_private_or_unverifiable_company_fact_rows",
    "build_repeat_exact_n_times_rows",
    "build_seed_rows",
    "materialize_seed_dataset",
    "read_jsonl",
    "validate_message",
    "validate_sft_row",
    "write_jsonl",
    "write_manifest",
]
