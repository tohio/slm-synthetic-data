"""Shared runtime primitives used by all five synthetic dataset pipelines."""

from .backend import build_backend
from .batching import (
    chunked,
    fill_exact_count,
    split_sequence_batch,
    split_slot_batch,
)
from .io import append_jsonl, reset_output_files, write_json, write_jsonl
from .novelty import NoveltyFilter, canonical_exact, jaccard, normalize, shingles
from .stages import Progress, run_model_stage_with_isolation
from .reporting import (
    build_deterministic_output_validation_summary,
    build_quality_decision_summary,
    deterministic_validation_blockers,
    estimate_dpo_tokens,
    estimate_sft_tokens,
    filter_validation_summary,
    quality_decision_blockers,
)

__all__ = [
    "Progress",
    "NoveltyFilter",
    "append_jsonl",
    "build_backend",
    "build_deterministic_output_validation_summary",
    "build_quality_decision_summary",
    "deterministic_validation_blockers",
    "estimate_dpo_tokens",
    "estimate_sft_tokens",
    "filter_validation_summary",
    "quality_decision_blockers",
    "canonical_exact",
    "chunked",
    "fill_exact_count",
    "jaccard",
    "normalize",
    "reset_output_files",
    "run_model_stage_with_isolation",
    "shingles",
    "split_sequence_batch",
    "split_slot_batch",
    "write_json",
    "write_jsonl",
]
