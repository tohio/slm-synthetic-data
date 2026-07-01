"""DPO synthetic row validation."""

from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import build_manifest_payload, write_manifest
from slm_synth.dpo.runs import DPOSeedRunResult, materialize_seed_dataset
from slm_synth.dpo.schema import validate_dpo_row, validate_message_list
from slm_synth.dpo.seeds import (
    DPO_SEED_FAMILIES,
    build_answer_only_arithmetic_rows,
    build_code_generation_function_rows,
    build_function_completion_body_only_rows,
    build_list_exact_n_items_rows,
    build_private_or_unverifiable_company_fact_rows,
    build_repeat_exact_n_times_rows,
    build_seed_rows,
)

__all__ = [
    "DPO_SEED_FAMILIES",
    "DPOSeedRunResult",
    "build_answer_only_arithmetic_rows",
    "build_code_generation_function_rows",
    "build_function_completion_body_only_rows",
    "build_list_exact_n_items_rows",
    "build_manifest_payload",
    "build_private_or_unverifiable_company_fact_rows",
    "build_repeat_exact_n_times_rows",
    "build_seed_rows",
    "materialize_seed_dataset",
    "read_jsonl",
    "validate_dpo_row",
    "validate_message_list",
    "write_jsonl",
    "write_manifest",
]
