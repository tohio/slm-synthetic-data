"""SFT synthetic row validation."""

from slm_synth.sft.io import read_jsonl, write_jsonl
from slm_synth.sft.manifest import build_manifest_payload, write_manifest
from slm_synth.sft.runs import SFTSeedRunResult, materialize_seed_dataset
from slm_synth.sft.schema import validate_message, validate_sft_row
from slm_synth.sft.seeds import SFT_SEED_FAMILIES, build_answer_only_arithmetic_rows, build_seed_rows

__all__ = [
    "SFT_SEED_FAMILIES",
    "SFTSeedRunResult",
    "build_manifest_payload",
    "build_answer_only_arithmetic_rows",
    "build_seed_rows",
    "materialize_seed_dataset",
    "read_jsonl",
    "validate_message",
    "validate_sft_row",
    "write_jsonl",
    "write_manifest",
]
