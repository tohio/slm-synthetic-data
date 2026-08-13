"""Deterministic response-diversity reporting for Distillation-SFT datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from collections import Counter
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.schema import validate_public_row


def normalize_response_text(value: str) -> str:
    """Normalize response text for exact diversity comparisons."""
    return " ".join(value.casefold().split())


def response_cluster_member_fingerprint(*, signal: str, row: Mapping[str, Any]) -> str:
    """Return a content-bound identifier for one cluster member."""
    payload = json.dumps(
        {"signal": signal, "row": dict(row)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def build_response_diversity_summary(files: Iterable[str | Path]) -> dict[str, Any]:
    """Build aggregate and per-signal exact response-diversity statistics."""
    counts_by_signal: dict[str, Counter[str]] = {}
    cluster_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    response_variants: dict[str, set[str]] = defaultdict(set)

    for raw_path in files:
        path = Path(raw_path)
        signal = path.stem.split(".batch", 1)[0]
        response_counts = counts_by_signal.setdefault(signal, Counter())
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
                row = validate_public_row(value)
                normalized_response = normalize_response_text(row["response"])
                response_counts[normalized_response] += 1
                response_variants[normalized_response].add(row["response"])
                cluster_members[normalized_response].append(
                    {
                        "id": row["id"],
                        "member_fingerprint": response_cluster_member_fingerprint(
                            signal=signal,
                            row=row,
                        ),
                        "prompt": row["prompt"],
                        "signal": signal,
                        "category": row["metadata"]["category"],
                        "template_family": row["metadata"]["template_family"],
                        "eval_family": row["metadata"]["eval_family"],
                    }
                )

    aggregate_counts: Counter[str] = Counter()
    signals: dict[str, dict[str, Any]] = {}
    for signal in sorted(counts_by_signal):
        response_counts = counts_by_signal[signal]
        aggregate_counts.update(response_counts)
        signals[signal] = _summarize_counts(response_counts)

    summary = _summarize_counts(aggregate_counts)
    summary["repeated_response_clusters"] = _build_repeated_response_clusters(
        aggregate_counts,
        cluster_members=cluster_members,
        response_variants=response_variants,
    )
    summary["signals"] = signals
    return summary


def _build_repeated_response_clusters(
    response_counts: Mapping[str, int],
    *,
    cluster_members: Mapping[str, list[dict[str, Any]]],
    response_variants: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for normalized_response, count in response_counts.items():
        if count <= 1:
            continue
        members = sorted(
            cluster_members[normalized_response],
            key=lambda member: (member["signal"], member["id"]),
        )
        clusters.append(
            {
                "response_fingerprint": sha256(normalized_response.encode("utf-8")).hexdigest(),
                "normalized_response": normalized_response,
                "responses": sorted(response_variants[normalized_response]),
                "count": count,
                "members": members,
            }
        )
    return sorted(
        clusters,
        key=lambda cluster: (-cluster["count"], cluster["response_fingerprint"]),
    )


def _summarize_counts(response_counts: Mapping[str, int]) -> dict[str, Any]:
    row_count = sum(response_counts.values())
    unique_response_count = len(response_counts)
    duplicate_response_count = row_count - unique_response_count
    repeated = sorted(
        (
            {"response": response[:160], "count": count}
            for response, count in response_counts.items()
            if count > 1
        ),
        key=lambda item: (-item["count"], item["response"]),
    )
    return {
        "row_count": row_count,
        "unique_response_count": unique_response_count,
        "duplicate_response_count": duplicate_response_count,
        "unique_response_ratio": unique_response_count / row_count if row_count else 0.0,
        "duplicate_examples": repeated[:10],
    }
