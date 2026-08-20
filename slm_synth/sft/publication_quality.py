"""Deterministic publication-quality diagnostics for accepted SFT rows."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from slm_synth.sft.acceptance import (
    normalize_sft_text,
    sft_conversation_fingerprint,
    sft_prompt_fingerprint,
    sft_response_fingerprint,
)
from slm_synth.sft.schema import validate_sft_row

NEAR_DUPLICATE_THRESHOLD = 0.88
MAX_TEMPLATE_SHARE = 0.40
MAX_REPORTED_PAIRS = 20
MAX_REPORTED_CLUSTERS = 20
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_publication_quality_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report near duplicates, repeated responses, and template concentration."""
    validated = [validate_sft_row(row) for row in rows]
    return {
        "near_duplicates": {
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "metric": "jaccard_normalized_token_set",
            "prompts": _near_duplicate_summary(validated, sft_prompt_fingerprint),
            "conversations": _near_duplicate_summary(validated, sft_conversation_fingerprint),
        },
        "assistant_response_clusters": _response_cluster_summary(validated),
        "template_concentration": _template_concentration_summary(validated),
    }


def _near_duplicate_summary(
    rows: list[dict[str, Any]], fingerprint: Callable[[Mapping[str, Any]], str]
) -> dict[str, Any]:
    values = [(row["id"], fingerprint(row)) for row in rows]
    tokens = [frozenset(_TOKEN_RE.findall(value)) for _, value in values]
    pairs: list[dict[str, Any]] = []
    involved: set[str] = set()
    pair_count = 0
    for left_index, (left_id, left_value) in enumerate(values):
        left_tokens = tokens[left_index]
        for right_index in range(left_index + 1, len(values)):
            right_id, right_value = values[right_index]
            if left_value == right_value:
                continue
            right_tokens = tokens[right_index]
            largest = max(len(left_tokens), len(right_tokens))
            if largest and min(len(left_tokens), len(right_tokens)) / largest < NEAR_DUPLICATE_THRESHOLD:
                continue
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity < NEAR_DUPLICATE_THRESHOLD:
                continue
            pair_count += 1
            involved.update((left_id, right_id))
            if len(pairs) < MAX_REPORTED_PAIRS:
                pairs.append(
                    {
                        "left_id": left_id,
                        "right_id": right_id,
                        "similarity": round(similarity, 6),
                    }
                )
    return {
        "pair_count": pair_count,
        "row_count": len(involved),
        "pairs": pairs,
    }


def _response_cluster_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    members: dict[str, list[str]] = defaultdict(list)
    variants: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        fingerprint = sft_response_fingerprint(row)
        members[fingerprint].append(row["id"])
        variants[fingerprint].add(
            "\n".join(
                normalize_sft_text(message.get("content") or json.dumps(message.get("tool_calls", []), sort_keys=True))
                for message in row["messages"]
                if message["role"] == "assistant"
            )
        )
    clusters = [
        {
            "count": len(ids),
            "row_ids": sorted(ids),
            "response_preview": sorted(variants[fingerprint])[0][:300],
        }
        for fingerprint, ids in members.items()
        if len(ids) > 1
    ]
    clusters.sort(key=lambda item: (-item["count"], item["row_ids"]))
    return {
        "cluster_count": len(clusters),
        "row_count": sum(item["count"] for item in clusters),
        "maximum_repetition": max((item["count"] for item in clusters), default=1 if rows else 0),
        "clusters": clusters[:MAX_REPORTED_CLUSTERS],
    }


def _template_concentration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["metadata"]["template_family"] for row in rows)
    total = len(rows)
    concentrated = [
        {
            "template_family": family,
            "count": count,
            "share": round(count / total, 6),
        }
        for family, count in sorted(counts.items())
        if count > 1 and count / total > MAX_TEMPLATE_SHARE
    ]
    concentrated.sort(key=lambda item: (-item["share"], item["template_family"]))
    maximum_count = max(counts.values(), default=0)
    return {
        "maximum_allowed_share": MAX_TEMPLATE_SHARE,
        "template_family_count": len(counts),
        "maximum_count": maximum_count,
        "maximum_share": round(maximum_count / total, 6) if total else 0.0,
        "concentrated_template_count": len(concentrated),
        "concentrated_templates": concentrated,
    }
