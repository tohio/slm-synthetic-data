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

__all__ = [
    "Progress",
    "NoveltyFilter",
    "append_jsonl",
    "build_backend",
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
