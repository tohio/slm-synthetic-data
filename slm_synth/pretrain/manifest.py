"""Run-level manifest helpers for synthetic pretraining data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from slm_synth.pretrain.grounded import GroundedBatchStore
from slm_synth.run_summary import print_pretrain_run_summary
from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE

PRETRAIN_STAGES = ("raw", "validated", "deduped", "rejected")


def build_run_manifest(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    generation_run: str | None = None,
    stages: tuple[str, ...] = PRETRAIN_STAGES,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only manifest summarizing pretrain stage outputs."""
    cfg = _load_yaml_config(config_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else _resolve_output_dir(cfg)
    run_name = _resolve_generation_run(cfg=cfg, output_dir=resolved_output_dir, generation_run=generation_run)
    stage_payloads = {
        stage: _summarize_stage(resolved_output_dir / stage)
        for stage in stages
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "pretrain",
        "generation_run": run_name,
        "config_path": str(Path(config_path)),
        "output_dir": str(resolved_output_dir),
        "stages": stage_payloads,
        "signals": _summarize_signals(stage_payloads),
        "metadata": {
            **_build_pretrain_metadata(resolved_output_dir),
            **dict(metadata or {}),
        },
    }


def write_run_manifest(
    *,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write a pretrain run manifest JSON file and return its path."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def default_manifest_path(
    *,
    output_dir: str | Path,
    generation_run: str,
) -> Path:
    """Return the default pretrain run-level manifest path."""
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(output_dir) / "manifests" / f"{generation_run.strip()}.manifest.json"


def generate_run_manifest(
    *,
    config_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    generation_run: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Build and write a pretrain run manifest from a config and existing outputs."""
    manifest = build_run_manifest(
        config_path=config_path,
        output_dir=output_dir,
        generation_run=generation_run,
        metadata=metadata,
    )
    manifest_path = Path(output) if output is not None else default_manifest_path(
        output_dir=manifest["output_dir"],
        generation_run=manifest["generation_run"],
    )
    return write_run_manifest(manifest_path=manifest_path, manifest=manifest)


def _resolve_generation_run(
    *,
    cfg: Mapping[str, Any],
    output_dir: Path,
    generation_run: str | None,
) -> str:
    if generation_run is not None and generation_run.strip():
        return generation_run.strip()
    configured = cfg.get("run_name") or cfg.get("run_id")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if output_dir.name:
        return output_dir.name
    raise ValueError("could not resolve generation_run from config or output_dir")


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pretrain config must contain a YAML mapping")
    return value


def _resolve_output_dir(cfg: Mapping[str, Any]) -> Path:
    raw = str(cfg.get("output_dir", "")).strip()
    if not raw:
        raise ValueError("pretrain config is missing required key: output_dir")

    os.environ.setdefault("DATA_DIR", "data/runs")
    return Path(os.path.expandvars(raw))


def _summarize_stage(stage_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    total_rows = 0
    jsonl_files = sorted(stage_dir.glob("*.jsonl")) if stage_dir.exists() else []

    for path in jsonl_files:
        rows = _count_jsonl_rows(path)
        signal = SIGNAL_FROM_FILE.get(path.name, path.stem)
        files[path.name] = {
            "signal": signal,
            "path": str(path),
            "rows": rows,
        }
        total_rows += rows

    return {
        "path": str(stage_dir),
        "exists": stage_dir.exists(),
        "file_count": len(jsonl_files),
        "row_count": total_rows,
        "files": files,
    }


def _summarize_signals(stage_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    signals: dict[str, dict[str, int]] = {}
    for stage, payload in stage_payloads.items():
        files = payload.get("files", {})
        if not isinstance(files, Mapping):
            continue
        for file_payload in files.values():
            if not isinstance(file_payload, Mapping):
                continue
            signal = file_payload.get("signal")
            rows = file_payload.get("rows")
            if isinstance(signal, str) and isinstance(rows, int):
                signals.setdefault(signal, {})[f"{stage}_rows"] = rows
    return {signal: signals[signal] for signal in sorted(signals)}


def _build_pretrain_metadata(output_dir: Path) -> dict[str, Any]:
    telemetry = _summarize_grounded_telemetry(output_dir)
    if not telemetry:
        return {}
    return {"telemetry": telemetry}


def _summarize_grounded_telemetry(output_dir: Path) -> dict[str, Any]:
    grounded_dir = output_dir / "manifests" / "grounded"
    if not grounded_dir.exists():
        return {}

    signals: dict[str, dict[str, Any]] = {}
    for signal_dir in sorted(path for path in grounded_dir.iterdir() if path.is_dir()):
        signal = signal_dir.name
        metrics = GroundedBatchStore(output_dir, signal).telemetry_summary()
        if int(metrics.get("batches", 0) or 0) or int(metrics.get("dropped_batches", 0) or 0):
            signals[signal] = dict(metrics)

    if not signals:
        return {}

    return {
        "signals": signals,
        "totals": _aggregate_grounded_telemetry(signals.values()),
    }


def _aggregate_grounded_telemetry(metrics_by_signal: Any) -> dict[str, Any]:
    totals = {
        "batches": 0,
        "dropped_batches": 0,
        "dropped_rows": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "aggregate_request_seconds": 0.0,
        "retry_count": 0,
        "retryable_provider_retries": 0,
        "retry_sleep_seconds": 0.0,
        "adaptive_window_increases": 0,
        "adaptive_window_decreases": 0,
        "adaptive_admission_wait_seconds": 0.0,
        "adaptive_peak_in_flight_limit": 0,
        "adaptive_min_in_flight_limit": 0,
        "max_adaptive_cooldown_seconds": 0.0,
        "adaptive_batch_size_observed_minimum": 0,
        "adaptive_batch_size_observed_peak": 0,
        "adaptive_batch_size_increases": 0,
        "adaptive_batch_size_decreases": 0,
        "adaptive_batch_size_failures": 0,
    }
    min_in_flight: int | None = None
    min_batch_size: int | None = None

    for metrics in metrics_by_signal:
        totals["batches"] += int(metrics.get("batches", 0) or 0)
        totals["dropped_batches"] += int(metrics.get("dropped_batches", 0) or 0)
        totals["dropped_rows"] += int(metrics.get("dropped_rows", 0) or 0)
        totals["prompt_tokens"] += int(metrics.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(metrics.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(metrics.get("total_tokens", 0) or 0)
        totals["cost"] += float(metrics.get("cost", 0.0) or 0.0)
        totals["aggregate_request_seconds"] += _request_seconds(metrics)
        totals["retry_count"] += int(metrics.get("retry_count", 0) or 0)
        totals["retryable_provider_retries"] += int(metrics.get("retryable_provider_retries", 0) or 0)
        totals["retry_sleep_seconds"] += float(metrics.get("retry_sleep_seconds", 0.0) or 0.0)
        totals["adaptive_window_increases"] += int(metrics.get("adaptive_window_increases", 0) or 0)
        totals["adaptive_window_decreases"] += int(metrics.get("adaptive_window_decreases", 0) or 0)
        totals["adaptive_admission_wait_seconds"] += float(
            metrics.get("adaptive_admission_wait_seconds", 0.0) or 0.0
        )
        totals["adaptive_peak_in_flight_limit"] = max(
            totals["adaptive_peak_in_flight_limit"],
            int(metrics.get("adaptive_peak_in_flight_limit", 0) or 0),
        )
        observed_min = int(metrics.get("adaptive_min_in_flight_limit", 0) or 0)
        if observed_min:
            min_in_flight = observed_min if min_in_flight is None else min(min_in_flight, observed_min)
        totals["max_adaptive_cooldown_seconds"] = max(
            totals["max_adaptive_cooldown_seconds"],
            float(metrics.get("max_adaptive_cooldown_seconds", 0.0) or 0.0),
        )
        observed_batch_min = int(metrics.get("adaptive_batch_size_observed_minimum", 0) or 0)
        if observed_batch_min:
            min_batch_size = observed_batch_min if min_batch_size is None else min(min_batch_size, observed_batch_min)
        totals["adaptive_batch_size_observed_peak"] = max(
            totals["adaptive_batch_size_observed_peak"],
            int(metrics.get("adaptive_batch_size_observed_peak", 0) or 0),
        )
        totals["adaptive_batch_size_increases"] += int(metrics.get("adaptive_batch_size_increases", 0) or 0)
        totals["adaptive_batch_size_decreases"] += int(metrics.get("adaptive_batch_size_decreases", 0) or 0)
        totals["adaptive_batch_size_failures"] += int(metrics.get("adaptive_batch_size_failures", 0) or 0)

    totals["cost"] = round(totals["cost"], 8)
    totals["aggregate_request_seconds"] = round(totals["aggregate_request_seconds"], 3)
    totals["retry_sleep_seconds"] = round(totals["retry_sleep_seconds"], 3)
    totals["adaptive_admission_wait_seconds"] = round(totals["adaptive_admission_wait_seconds"], 3)
    totals["max_adaptive_cooldown_seconds"] = round(totals["max_adaptive_cooldown_seconds"], 3)
    totals["adaptive_min_in_flight_limit"] = min_in_flight or 0
    totals["adaptive_batch_size_observed_minimum"] = min_batch_size or 0
    return totals


def _request_seconds(metrics: Mapping[str, Any]) -> float:
    if "aggregate_request_seconds" in metrics:
        return float(metrics.get("aggregate_request_seconds", 0.0) or 0.0)
    return float(metrics.get("elapsed_seconds", 0.0) or 0.0)


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a pretrain run-level manifest from existing outputs.")
    parser.add_argument("--config", required=True, help="Path to configs/synthetic.yaml")
    parser.add_argument("--output", default=None, help="Optional manifest output path")
    parser.add_argument("--output-dir", default=None, help="Optional generated-output directory override")
    parser.add_argument("--generation-run", default=None, help="Optional generation run name override")
    args = parser.parse_args(argv)

    path = generate_run_manifest(
        config_path=args.config,
        output=args.output,
        output_dir=args.output_dir,
        generation_run=args.generation_run,
    )
    print(f"wrote pretrain run manifest to {path}")
    print_pretrain_run_summary(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
