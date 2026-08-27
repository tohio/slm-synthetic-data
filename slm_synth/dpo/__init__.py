"""DPO synthetic row validation."""

from slm_synth.dpo.batches import (
    build_dpo_teacher_request_items,
    build_dpo_teacher_request_object,
    validate_dpo_batch_response,
)
from slm_synth.dpo.generation import (
    DPOLLMBatchResult,
    build_openrouter_backend,
    generate_llm_batch,
    materialize_llm_batch,
)
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import build_manifest_payload, write_manifest, write_run_manifest
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.dpo.schema import validate_dpo_row, validate_message_list
from slm_synth.dpo.spec_builders import (
    DPO_PREFERENCE_DIMENSIONS,
    build_specs,
)
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec

__all__ = [
    "DPO_PREFERENCE_DIMENSIONS",
    "DPOLLMBatchResult",
    "build_coverage_report",
    "build_manifest_payload",
    "build_specs",
    "build_openrouter_backend",
    "build_dpo_teacher_request_items",
    "build_dpo_teacher_request_object",
    "generate_llm_batch",
    "materialize_llm_batch",
    "read_jsonl",
    "teacher_visible_dpo_spec",
    "validate_dpo_batch_response",
    "validate_dpo_row",
    "validate_dpo_spec",
    "validate_message_list",
    "write_jsonl",
    "write_coverage_report",
    "write_manifest",
    "write_run_manifest",
]
