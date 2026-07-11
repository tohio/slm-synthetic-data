"""Human-readable run summary helpers for generation CLIs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def print_pretrain_run_summary(manifest_path: str | Path) -> None:
    manifest = _load_json(manifest_path)
    metadata = _metadata(manifest)
    telemetry = _pretrain_telemetry(metadata)
    rows = _pretrain_rows(manifest)
    signals = manifest.get("signals", {})
    signal_count = len(signals) if isinstance(signals, Mapping) else 0
    print(
        "[generate] Completed pretrain run: "
        f"rows={rows.get('deduped', 0)}, "
        f"raw_rows={rows.get('raw', 0)}, "
        f"validated_rows={rows.get('validated', 0)}, "
        f"rejected_rows={rows.get('rejected', 0)}, "
        f"signals={signal_count}, "
        f"adaptive_batch_size_observed_minimum={telemetry.get('adaptive_batch_size_observed_minimum', 'n/a')}, "
        f"adaptive_batch_size_observed_peak={telemetry.get('adaptive_batch_size_observed_peak', 'n/a')}, "
        f"adaptive_batch_size_increases={telemetry.get('adaptive_batch_size_increases', 'n/a')}, "
        f"adaptive_batch_size_decreases={telemetry.get('adaptive_batch_size_decreases', 'n/a')}, "
        f"adaptive_batch_size_failures={telemetry.get('adaptive_batch_size_failures', 'n/a')}, "
        f"{_pretrain_telemetry_text(telemetry)}"
    )
    print(f"[generate] pretrain signals={json.dumps(_pretrain_signal_rows(manifest), sort_keys=True)}")


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


def print_batch_progress(
    *,
    workflow: str,
    group_key: str,
    group_value: str,
    batch_number: int,
    batch_start: int,
    batch_size: int,
    rows_done: int,
    rows_total: int,
    manifest_path: str | Path,
    adaptive_batch_size: Mapping[str, Any],
) -> None:
    manifest = _load_json(manifest_path)
    metadata = _metadata(manifest)
    telemetry = _llm_telemetry(metadata)
    usage = telemetry.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    print(
        f"[generate] {workflow}: "
        f"{group_key}={group_value} "
        f"batch={batch_number} "
        f"batch_start={batch_start} "
        f"batch_size={batch_size} "
        f"rows={rows_done}/{rows_total} "
        f"adaptive_batch_size_current={adaptive_batch_size.get('adaptive_batch_size_current', 'n/a')} "
        f"adaptive_batch_size_observed_minimum={adaptive_batch_size.get('adaptive_batch_size_observed_minimum', 'n/a')} "
        f"adaptive_batch_size_observed_peak={adaptive_batch_size.get('adaptive_batch_size_observed_peak', 'n/a')} "
        f"adaptive_batch_size_increases={adaptive_batch_size.get('adaptive_batch_size_increases', 'n/a')} "
        f"adaptive_batch_size_decreases={adaptive_batch_size.get('adaptive_batch_size_decreases', 'n/a')} "
        f"adaptive_batch_size_failures={adaptive_batch_size.get('adaptive_batch_size_failures', 'n/a')} "
        f"provider_retries={int(telemetry.get('retryable_provider_retries', 0) or 0)} "
        f"retry_sleep_seconds={float(telemetry.get('retry_sleep_seconds', 0.0) or 0.0):.3f} "
        f"adaptive_peak_in_flight_limit={int(telemetry.get('adaptive_peak_in_flight_limit', 0) or 0)} "
        f"cost={float(usage.get('cost', 0.0) or 0.0):.8f} "
        f"request_tokens={int(usage.get('total_tokens', 0) or 0)}",
        flush=True,
    )


def print_batch_failure(
    *,
    workflow: str,
    group_key: str,
    group_value: str,
    batch_number: int,
    batch_start: int,
    batch_size: int,
    adaptive_batch_size: Mapping[str, Any],
    error: BaseException,
) -> None:
    print(
        f"[generate] {workflow}: "
        f"{group_key}={group_value} "
        f"batch={batch_number} "
        f"batch_start={batch_start} "
        f"batch_size={batch_size} "
        f"retrying_after_failure=true "
        f"adaptive_batch_size_current={adaptive_batch_size.get('adaptive_batch_size_current', 'n/a')} "
        f"adaptive_batch_size_observed_minimum={adaptive_batch_size.get('adaptive_batch_size_observed_minimum', 'n/a')} "
        f"adaptive_batch_size_decreases={adaptive_batch_size.get('adaptive_batch_size_decreases', 'n/a')} "
        f"adaptive_batch_size_failures={adaptive_batch_size.get('adaptive_batch_size_failures', 'n/a')} "
        f"error={str(error)!r}",
        flush=True,
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


def _pretrain_rows(manifest: Mapping[str, Any]) -> dict[str, int]:
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        return {"raw": 0, "validated": 0, "deduped": 0, "rejected": 0}
    rows: dict[str, int] = {}
    for stage in ("raw", "validated", "deduped", "rejected"):
        payload = stages.get(stage, {})
        rows[stage] = int(payload.get("row_count", 0) or 0) if isinstance(payload, Mapping) else 0
    return rows


def _pretrain_signal_rows(manifest: Mapping[str, Any]) -> dict[str, int]:
    signals = manifest.get("signals", {})
    if not isinstance(signals, Mapping):
        return {}
    rows: dict[str, int] = {}
    for name, payload in signals.items():
        if isinstance(name, str) and isinstance(payload, Mapping):
            rows[name] = int(payload.get("deduped_rows", payload.get("validated_rows", payload.get("raw_rows", 0))) or 0)
    return dict(sorted(rows.items()))


def _pretrain_telemetry(metadata: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = metadata.get("telemetry", {})
    if not isinstance(telemetry, Mapping):
        return {}
    totals = telemetry.get("totals", {})
    return dict(totals) if isinstance(totals, Mapping) else {}


def _pretrain_telemetry_text(telemetry: Mapping[str, Any]) -> str:
    return (
        f"batches={int(telemetry.get('batches', 0) or 0)}, "
        f"retry_count={int(telemetry.get('retry_count', 0) or 0)}, "
        f"provider_retries={int(telemetry.get('retryable_provider_retries', 0) or 0)}, "
        f"retry_sleep_seconds={float(telemetry.get('retry_sleep_seconds', 0.0) or 0.0):.3f}, "
        f"adaptive_window_increases={int(telemetry.get('adaptive_window_increases', 0) or 0)}, "
        f"adaptive_window_decreases={int(telemetry.get('adaptive_window_decreases', 0) or 0)}, "
        f"adaptive_admission_wait_seconds={float(telemetry.get('adaptive_admission_wait_seconds', 0.0) or 0.0):.3f}, "
        f"adaptive_peak_in_flight_limit={int(telemetry.get('adaptive_peak_in_flight_limit', 0) or 0)}, "
        f"adaptive_min_in_flight_limit={int(telemetry.get('adaptive_min_in_flight_limit', 0) or 0)}, "
        f"max_adaptive_cooldown_seconds={float(telemetry.get('max_adaptive_cooldown_seconds', 0.0) or 0.0):.3f}, "
        f"cost={float(telemetry.get('cost', 0.0) or 0.0):.8f}, "
        f"request_tokens={int(telemetry.get('total_tokens', 0) or 0)}, "
        f"aggregate_request_seconds={_aggregate_request_seconds(telemetry):.3f}"
    )


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
        f"aggregate_request_seconds={_aggregate_request_seconds(telemetry):.3f}"
    )


def _aggregate_request_seconds(telemetry: Mapping[str, Any]) -> float:
    if "aggregate_request_seconds" in telemetry:
        return float(telemetry.get("aggregate_request_seconds", 0.0) or 0.0)
    return float(telemetry.get("elapsed_seconds", 0.0) or 0.0)


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
