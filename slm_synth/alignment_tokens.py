"""Tokenizer-independent size estimates for public alignment artifacts."""

from __future__ import annotations

import json
import math
from typing import Any

DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_sft_tokens(row: dict[str, Any], *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate tokens in the public SFT training payload, excluding audit metadata."""
    payload = {"messages": row["messages"]}
    if "tools" in row:
        payload["tools"] = row["tools"]
    return _estimate_payload(payload, chars_per_token=chars_per_token)


def estimate_dpo_tokens(row: dict[str, Any], *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate tokens in the public DPO pair payload, excluding audit metadata."""
    payload = {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }
    if "tools" in row:
        payload["tools"] = row["tools"]
    return _estimate_payload(payload, chars_per_token=chars_per_token)


def _estimate_payload(payload: dict[str, Any], *, chars_per_token: float) -> int:
    if not isinstance(chars_per_token, (int, float)) or isinstance(chars_per_token, bool) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return max(1, math.ceil(len(serialized) / float(chars_per_token)))
