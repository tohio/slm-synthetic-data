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
    build_openrouter_backend,
    generate_llm_batch,
    generate_llm_batch_from_files,
    generate_teacher_batch_response,
    materialize_llm_batch,
    materialize_llm_batch_from_files,
    read_specs_jsonl,
    read_teacher_response_json,
)
from slm_synth.sft.io import read_jsonl, write_jsonl
from slm_synth.sft.manifest import build_manifest_payload, write_manifest, write_run_manifest
from slm_synth.sft.report import build_coverage_report, write_coverage_report
from slm_synth.sft.runs import (
    SFTLLMRunResult,
    generate_llm_run,
    resolve_spec_families,
)
from slm_synth.sft.schema import validate_message, validate_sft_row
from slm_synth.sft.spec_builders import (
    SFT_SPEC_FAMILIES,
    build_and_write_specs,
    build_specs,
    write_specs_jsonl,
)
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec

__all__ = [
    "SFT_BATCH_RESPONSE_SCHEMA",
    "SFTLLMRunResult",
    "SFT_SPEC_FAMILIES",
    "SFTLLMBatchResult",
    "build_manifest_payload",
    "build_coverage_report",
    "build_and_write_specs",
    "build_specs",
    "build_sft_teacher_request_items",
    "build_sft_teacher_request_object",
    "build_openrouter_backend",
    "generate_llm_batch",
    "generate_llm_batch_from_files",
    "generate_llm_run",
    "generate_teacher_batch_response",
    "materialize_llm_batch",
    "materialize_llm_batch_from_files",
    "read_jsonl",
    "read_specs_jsonl",
    "read_teacher_response_json",
    "render_sft_batch_prompt",
    "resolve_spec_families",
    "teacher_visible_sft_spec",
    "validate_message",
    "validate_sft_batch_response",
    "validate_sft_row",
    "validate_sft_spec",
    "write_jsonl",
    "write_specs_jsonl",
    "write_coverage_report",
    "write_manifest",
    "write_run_manifest",
]
