"""Human-readable run summary helpers for generation CLIs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def print_sft_run_summary(manifest_path: str | Path) -> None:
    manifest = _load_json(manifest_path)
    _print_chat_run_summary(label="SFT", manifest=manifest)


def print_dpo_run_summary(manifest_path: str | Path) -> None:
    manifest = _load_json(manifest_path)
    _print_chat_run_summary(label="DPO", manifest=manifest)


def print_distillation_run_summary(manifest_path: str | Path) -> None:
    manifest = _load_json(manifest_path)
    metadata = _metadata(manifest)
    telemetry = _llm_telemetry(metadata)
    signals = _count_by_key(manifest.get("datasets", []), "signal")
    print(
        "[generate] Completed distillation run: "
        f"rows={int(manifest.get('total_rows', 0) or 0)}, "
        f"signals={len(signals)}, "
        f"batch_size={metadata.get('batch_size', 'n/a')}, "
        f"concurrency={metadata.get('concurrency', 'n/a')}, "
        f"adaptive_maximum_in_flight={metadata.get('adaptive_maximum_in_flight', 'n/a')}, "
        f"adaptive_initial_in_flight={metadata.get('adaptive_initial_in_flight', 'n/a')}, "
        f"{_telemetry_text(telemetry)}"
    )
    print(f"[generate] distillation signals={json.dumps(signals, sort_keys=True)}")

    for dataset in manifest.get("datasets", []):
        if not isinstance(dataset, Mapping):
            continue
        signal_manifest_path = dataset.get("manifest_path")
        if not signal_manifest_path:
            continue
        signal_manifest = _load_json(signal_manifest_path)
        signal_metadata = _metadata(signal_manifest)
        signal_telemetry = _llm_telemetry(signal_metadata)
        print(
            "[generate] Completed distillation signal: "
            f"{signal_manifest.get('signal', dataset.get('signal', 'unknown'))} "
            f"rows={int(signal_manifest.get('row_count', 0) or 0)}, "
            f"batch_size={signal_metadata.get('batch_size', 'n/a')}, "
            f"concurrency={signal_metadata.get('concurrency', 'n/a')}, "
            f"adaptive_batch_size_observed_minimum={signal_metadata.get('adaptive_batch_size_observed_minimum', 'n/a')}, "
            f"adaptive_batch_size_observed_peak={signal_metadata.get('adaptive_batch_size_observed_peak', 'n/a')}, "
            f"adaptive_batch_size_increases={signal_metadata.get('adaptive_batch_size_increases', 'n/a')}, "
            f"adaptive_batch_size_decreases={signal_metadata.get('adaptive_batch_size_decreases', 'n/a')}, "
            f"adaptive_batch_size_failures={signal_metadata.get('adaptive_batch_size_failures', 'n/a')}, "
            f"{_telemetry_text(signal_telemetry)}"
        )


def _print_chat_run_summary(*, label: str, manifest: Mapping[str, Any]) -> None:
    metadata = _metadata(manifest)
    telemetry = _llm_telemetry(metadata)
    families = _count_by_key(manifest.get("datasets", []), "family")
    print(
        f"[generate] Completed {label} run: "
        f"rows={int(manifest.get('total_rows', 0) or 0)}, "
        f"families={len(families)}, "
        f"batch_size={metadata.get('batch_size', 'n/a')}, "
        f"concurrency={metadata.get('concurrency', 'n/a')}, "
        f"adaptive_maximum_in_flight={metadata.get('adaptive_maximum_in_flight', 'n/a')}, "
        f"adaptive_initial_in_flight={metadata.get('adaptive_initial_in_flight', 'n/a')}, "
        f"adaptive_batch_size_observed_minimum={metadata.get('adaptive_batch_size_observed_minimum', 'n/a')}, "
        f"adaptive_batch_size_observed_peak={metadata.get('adaptive_batch_size_observed_peak', 'n/a')}, "
        f"adaptive_batch_size_increases={metadata.get('adaptive_batch_size_increases', 'n/a')}, "
        f"adaptive_batch_size_decreases={metadata.get('adaptive_batch_size_decreases', 'n/a')}, "
        f"adaptive_batch_size_failures={metadata.get('adaptive_batch_size_failures', 'n/a')}, "
        f"{_telemetry_text(telemetry)}"
    )
    print(f"[generate] {label} families={json.dumps(families, sort_keys=True)}")


def _telemetry_text(telemetry: Mapping[str, Any]) -> str:
    usage = telemetry.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    return (
        f"batches={int(telemetry.get('batch_count', 0) or 0)}, "
        f"retry_count={int(telemetry.get('retry_count', 0) or 0)}, "
        f"provider_retries={int(telemetry.get('retryable_provider_retries', 0) or 0)}, "
        f"retry_sleep_seconds={float(telemetry.get('retry_sleep_seconds', 0.0) or 0.0):.3f}, "
        f"adaptive_window_increases={int(telemetry.get('adaptive_window_increases', 0) or 0)}, "
        f"adaptive_window_decreases={int(telemetry.get('adaptive_window_decreases', 0) or 0)}, "
        f"adaptive_admission_wait_seconds={float(telemetry.get('adaptive_admission_wait_seconds', 0.0) or 0.0):.3f}, "
        f"adaptive_peak_in_flight_limit={int(telemetry.get('adaptive_peak_in_flight_limit', 0) or 0)}, "
        f"adaptive_min_in_flight_limit={int(telemetry.get('adaptive_min_in_flight_limit', 0) or 0)}, "
        f"max_adaptive_cooldown_seconds={float(telemetry.get('max_adaptive_cooldown_seconds', 0.0) or 0.0):.3f}, "
        f"cost={float(usage.get('cost', 0.0) or 0.0):.8f}, "
        f"request_tokens={int(usage.get('total_tokens', 0) or 0)}, "
        f"aggregate_request_seconds={float(telemetry.get('elapsed_seconds', 0.0) or 0.0):.3f}"
    )


def _count_by_key(items: Any, key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get(key)
        rows = item.get("row_count")
        if isinstance(name, str) and isinstance(rows, int):
            counts[name] += rows
    return dict(sorted(counts.items()))


def _metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _llm_telemetry(metadata: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = metadata.get("llm_telemetry", {})
    return dict(telemetry) if isinstance(telemetry, Mapping) else {}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value
