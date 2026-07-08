"""Shared manifest telemetry aggregation helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def aggregate_llm_telemetry_from_manifests(manifest_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Aggregate per-batch LLM telemetry from local manifest files."""
    telemetry_items: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        metadata = _read_manifest_metadata(Path(manifest_path))
        telemetry = metadata.get("llm_telemetry")
        if isinstance(telemetry, Mapping):
            telemetry_items.append(dict(telemetry))
    return aggregate_llm_telemetry(telemetry_items)


def aggregate_llm_telemetry(telemetry_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return aggregate-safe telemetry fields for a set of provider calls."""
    if not telemetry_items:
        return {}

    merged: dict[str, Any] = {
        "batch_count": sum(_telemetry_batch_count(item) for item in telemetry_items),
        "usage": _sum_usage(telemetry_items),
        "retry_count": sum(int(item.get("retry_count", 0) or 0) for item in telemetry_items),
        "retryable_provider_retries": sum(
            int(item.get("retryable_provider_retries", 0) or 0) for item in telemetry_items
        ),
        "retry_sleep_seconds": round(
            sum(float(item.get("retry_sleep_seconds", 0.0) or 0.0) for item in telemetry_items),
            3,
        ),
        "adaptive_window_increases": sum(
            int(item.get("adaptive_window_increases", 0) or 0) for item in telemetry_items
        ),
        "adaptive_window_decreases": sum(
            int(item.get("adaptive_window_decreases", 0) or 0) for item in telemetry_items
        ),
        "adaptive_admission_wait_seconds": round(
            sum(float(item.get("adaptive_admission_wait_seconds", 0.0) or 0.0) for item in telemetry_items),
            3,
        ),
        "adaptive_peak_in_flight_limit": max(
            int(item.get("adaptive_peak_in_flight_limit", 0) or 0) for item in telemetry_items
        ),
        "adaptive_min_in_flight_limit": _min_positive(
            int(item.get("adaptive_min_in_flight_limit", 0) or 0) for item in telemetry_items
        ),
        "max_adaptive_cooldown_seconds": round(
            max(float(item.get("max_adaptive_cooldown_seconds", 0.0) or 0.0) for item in telemetry_items),
            3,
        ),
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in telemetry_items), 3),
    }

    first = telemetry_items[0]
    for key in ("model", "provider", "routing_mode", "requested_provider", "allow_fallbacks"):
        if key in first:
            merged[key] = first[key]
    return merged


def _read_manifest_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _sum_usage(telemetry_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    for item in telemetry_items:
        item_usage = item.get("usage", {})
        if not isinstance(item_usage, Mapping):
            continue
        usage["prompt_tokens"] += int(item_usage.get("prompt_tokens", 0) or 0)
        usage["completion_tokens"] += int(item_usage.get("completion_tokens", 0) or 0)
        usage["total_tokens"] += int(item_usage.get("total_tokens", 0) or 0)
        usage["cost"] += float(item_usage.get("cost", 0.0) or 0.0)
    usage["cost"] = round(usage["cost"], 8)
    return usage


def _min_positive(values: Iterable[int]) -> int:
    positive_values = [value for value in values if value > 0]
    return min(positive_values) if positive_values else 0


def _telemetry_batch_count(item: Mapping[str, Any]) -> int:
    """Return the represented request/batch count for raw or nested telemetry.

    Raw provider-call telemetry normally has no batch_count and represents one
    request. Aggregated signal-level telemetry carries batch_count and represents
    that many lower-level requests.
    """
    value = item.get("batch_count")
    if isinstance(value, int) and value > 0:
        return value
    return 1
