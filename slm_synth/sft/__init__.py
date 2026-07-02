"""SFT synthetic row validation."""

from slm_synth.sft.batches import (
    SFT_BATCH_RESPONSE_SCHEMA,
    build_sft_teacher_request_items,
    build_sft_teacher_request_object,
    render_sft_batch_prompt,
    validate_sft_batch_response,
)
from slm_synth.sft.generation import (
    SFTLLMBatchResult,
    materialize_llm_batch,
    materialize_llm_batch_from_files,
    read_specs_jsonl,
    read_teacher_response_json,
)
from slm_synth.sft.io import read_jsonl, write_jsonl
from slm_synth.sft.manifest import build_manifest_payload, write_manifest, write_run_manifest
from slm_synth.sft.report import build_coverage_report, write_coverage_report
from slm_synth.sft.runs import (
    SFTSeedFamilyRunResult,
    SFTSeedRunResult,
    materialize_seed_dataset,
    materialize_seed_run,
    resolve_seed_families,
)
from slm_synth.sft.schema import validate_message, validate_sft_row
from slm_synth.sft.seeds import (
    SFT_SEED_FAMILIES,
    build_ai_concept_explanation_rows,
    build_answer_only_arithmetic_rows,
    build_capital_city_qa_rows,
    build_clear_sky_color_qa_rows,
    build_code_explanation_no_code_rows,
    build_code_expression_result_rows,
    build_code_generation_function_rows,
    build_direct_division_rows,
    build_direct_subtraction_rows,
    build_function_completion_body_only_rows,
    build_list_exact_n_items_rows,
    build_private_or_unverifiable_company_fact_rows,
    build_repeat_exact_n_times_rows,
    build_seed_rows,
)
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec

__all__ = [
    "SFT_BATCH_RESPONSE_SCHEMA",
    "SFTLLMBatchResult",
    "SFT_SEED_FAMILIES",
    "SFTSeedFamilyRunResult",
    "SFTSeedRunResult",
    "build_ai_concept_explanation_rows",
    "build_manifest_payload",
    "build_answer_only_arithmetic_rows",
    "build_capital_city_qa_rows",
    "build_coverage_report",
    "build_clear_sky_color_qa_rows",
    "build_code_explanation_no_code_rows",
    "build_code_expression_result_rows",
    "build_code_generation_function_rows",
    "build_direct_division_rows",
    "build_direct_subtraction_rows",
    "build_function_completion_body_only_rows",
    "build_list_exact_n_items_rows",
    "build_private_or_unverifiable_company_fact_rows",
    "build_repeat_exact_n_times_rows",
    "build_seed_rows",
    "build_sft_teacher_request_items",
    "build_sft_teacher_request_object",
    "materialize_seed_dataset",
    "materialize_seed_run",
    "materialize_llm_batch",
    "materialize_llm_batch_from_files",
    "read_jsonl",
    "read_specs_jsonl",
    "read_teacher_response_json",
    "render_sft_batch_prompt",
    "resolve_seed_families",
    "teacher_visible_sft_spec",
    "validate_message",
    "validate_sft_batch_response",
    "validate_sft_row",
    "validate_sft_spec",
    "write_jsonl",
    "write_coverage_report",
    "write_manifest",
    "write_run_manifest",
]
