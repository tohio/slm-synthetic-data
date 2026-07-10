"""DPO synthetic row validation."""

from slm_synth.dpo.batches import (
    DPO_BATCH_RESPONSE_SCHEMA,
    build_dpo_teacher_request_items,
    build_dpo_teacher_request_object,
    render_dpo_batch_prompt,
    validate_dpo_batch_response,
)
from slm_synth.dpo.generation import (
    DPOLLMBatchResult,
    build_openrouter_backend,
    generate_llm_batch,
    generate_llm_batch_from_files,
    generate_teacher_batch_response,
    materialize_llm_batch,
    materialize_llm_batch_from_files,
    read_specs_jsonl,
    read_teacher_response_json,
)
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import build_manifest_payload, write_manifest, write_run_manifest
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.dpo.runs import (
    DPOLLMRunResult,
    generate_llm_run,
    resolve_spec_families,
)
from slm_synth.dpo.schema import validate_dpo_row, validate_message_list
from slm_synth.dpo.spec_builders import (
    DPO_SPEC_FAMILIES,
    build_and_write_specs,
    build_specs,
    write_specs_jsonl,
)
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec

__all__ = [
    "DPO_BATCH_RESPONSE_SCHEMA",
    "DPOLLMRunResult",
    "DPO_SPEC_FAMILIES",
    "DPOLLMBatchResult",
    "build_coverage_report",
    "build_manifest_payload",
    "build_and_write_specs",
    "build_specs",
    "build_openrouter_backend",
    "build_dpo_teacher_request_items",
    "build_dpo_teacher_request_object",
    "generate_llm_batch",
    "generate_llm_batch_from_files",
    "generate_llm_run",
    "generate_teacher_batch_response",
    "materialize_llm_batch",
    "materialize_llm_batch_from_files",
    "read_jsonl",
    "read_specs_jsonl",
    "read_teacher_response_json",
    "render_dpo_batch_prompt",
    "resolve_spec_families",
    "teacher_visible_dpo_spec",
    "validate_dpo_batch_response",
    "validate_dpo_row",
    "validate_dpo_spec",
    "validate_message_list",
    "write_jsonl",
    "write_specs_jsonl",
    "write_coverage_report",
    "write_manifest",
    "write_run_manifest",
]
