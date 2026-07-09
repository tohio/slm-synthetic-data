#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.llm import LLMBackend
from slm_synth.model_support import warn_if_unsupported_model
from slm_synth.pretrain.writer import JSONLWriter
from slm_synth.pretrain.grounded import (
    GroundedBatchStore,
    GroundedRenderedBatchError,
    GroundedSignalGenerator,
    GroundedTransientProviderBatchError,
)

load_dotenv()


MIN_GROUNDED_BATCH_SIZE = 1
MAX_GROUNDED_BATCH_SIZE = 64
MIN_GROUNDED_PARALLEL_REQUESTS = 1
MAX_GROUNDED_PARALLEL_REQUESTS = 1024


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _int_cfg(*values: Any, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _float_cfg(*values: Any, default: float) -> float:
    for value in values:
        if value is not None:
            return float(value)
    return float(default)


def _bool_cfg(*values: Any, default: bool) -> bool:
    for value in values:
        if value is not None:
            return bool(value)
    return bool(default)


def build_llm(
    base_cfg: Dict[str, Any],
    signal_cfg: Optional[Dict[str, Any]] = None,
    *,
    role: str = "candidate",
) -> LLMBackend:
    signal_cfg = signal_cfg or {}
    retry_cfg = base_cfg.get("retries", {}) or {}

    model_name = signal_cfg.get(
        f"{role}_model",
        signal_cfg.get("model", base_cfg["model"]),
    )
    warn_if_unsupported_model(model_name, context=f"synthetic {role} generation")

    return LLMBackend(
        provider=base_cfg.get("provider", "openrouter"),
        model=model_name,
        max_tokens=_int_cfg(signal_cfg.get("max_tokens"), base_cfg.get("max_tokens"), default=1024),
        temperature=_float_cfg(signal_cfg.get("temperature"), base_cfg.get("temperature"), default=0.2),
        top_p=_float_cfg(signal_cfg.get("top_p"), base_cfg.get("top_p"), default=0.95),
        json_mode=_bool_cfg(signal_cfg.get("json_mode"), base_cfg.get("json_mode"), default=True),
        service_tier=signal_cfg.get("service_tier", base_cfg.get("service_tier")),
        request_timeout=base_cfg.get("request_timeout_seconds"),
        max_request_retries=int(retry_cfg.get("max_request_retries", 3)),
        max_retryable_request_attempts=int(retry_cfg.get("max_retryable_request_attempts", 20)),
        retry_max_elapsed_seconds=float(retry_cfg.get("retry_max_elapsed_seconds", 1800.0)),
        retry_sleep_seconds=float(retry_cfg.get("retry_sleep_seconds", 0.5)),
        retry_backoff_initial_seconds=float(retry_cfg.get("retry_backoff_initial_seconds", 1.0)),
        retry_backoff_max_seconds=float(retry_cfg.get("retry_backoff_max_seconds", 30.0)),
        retry_backoff_multiplier=float(retry_cfg.get("retry_backoff_multiplier", 2.0)),
        retry_jitter_ratio=float(retry_cfg.get("retry_jitter_ratio", 0.30)),
        adaptive_concurrency_enabled=bool(retry_cfg.get("adaptive_concurrency_enabled", True)),
        adaptive_maximum_in_flight=int(base_cfg.get("parallel_requests", 1)),
        adaptive_initial_in_flight=int(retry_cfg.get("adaptive_initial_in_flight", 8)),
        adaptive_minimum_in_flight=int(retry_cfg.get("adaptive_minimum_in_flight", 1)),
        adaptive_slow_start_enabled=bool(retry_cfg.get("adaptive_slow_start_enabled", True)),
        adaptive_slow_start_multiplier=float(retry_cfg.get("adaptive_slow_start_multiplier", 2.0)),
        adaptive_increase_successes_per_step=int(retry_cfg.get("adaptive_increase_successes_per_step", 64)),
        adaptive_increase_step=int(retry_cfg.get("adaptive_increase_step", 16)),
        adaptive_rate_limit_burst_threshold=int(retry_cfg.get("adaptive_rate_limit_burst_threshold", 4)),
        adaptive_rate_limit_window_seconds=float(retry_cfg.get("adaptive_rate_limit_window_seconds", 2.0)),
        adaptive_rate_limit_decrease_factor=float(retry_cfg.get("adaptive_rate_limit_decrease_factor", 0.50)),
        adaptive_sustained_rate_limit_attempt_window=int(retry_cfg.get("adaptive_sustained_rate_limit_attempt_window", 60)),
        adaptive_sustained_rate_limit_threshold=int(retry_cfg.get("adaptive_sustained_rate_limit_threshold", 20)),
        adaptive_cooldown_initial_seconds=float(retry_cfg.get("adaptive_cooldown_initial_seconds", 5.0)),
        adaptive_cooldown_max_seconds=float(retry_cfg.get("adaptive_cooldown_max_seconds", 60.0)),
        adaptive_cooldown_multiplier=float(retry_cfg.get("adaptive_cooldown_multiplier", 2.0)),
        require_parameters=bool(base_cfg.get("require_parameters", True)),
        allow_fallbacks=bool(base_cfg.get("allow_fallbacks", False)),
    )


def _grounded_token_target(cfg: Dict[str, Any], mix_cfg: Dict[str, Any]) -> int:
    explicit = mix_cfg.get("target_tokens")
    if explicit is not None:
        return int(explicit)
    return max(1, int(round(int(cfg["target_total_tokens"]) * float(mix_cfg.get("share", 0.0)))))


def _rounded_batch_target_rows(cfg: Dict[str, Any], mix_cfg: Dict[str, Any], batch_size: int) -> tuple[int, int, int]:
    generation_cfg = cfg.get("generation", {}) or {}
    explicit_rows = mix_cfg.get("samples")
    token_target = _grounded_token_target(cfg, mix_cfg)
    if explicit_rows is not None:
        target_rows = int(explicit_rows)
    else:
        avg_tokens = int(mix_cfg.get("avg_tokens_per_sample", generation_cfg.get("avg_tokens_per_sample", 80)))
        target_rows = max(1, (token_target + avg_tokens - 1) // avg_tokens)
    rounded_rows = ((target_rows + batch_size - 1) // batch_size) * batch_size
    return token_target, target_rows, rounded_rows


def _retry_grounded_batch_after_delay(
    generator: GroundedSignalGenerator,
    start_index: int,
    batch_size: int,
    delay_seconds: float,
):
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return generator.generate_range(start_index, batch_size, batch_id=start_index)


def _pending_grounded_ranges(rounded_rows: int, terminal_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    covered: list[tuple[int, int]] = []
    for start, size in terminal_ranges:
        end = min(rounded_rows, start + size)
        if start < rounded_rows and end > start:
            covered.append((start, end))
    covered.sort()

    pending: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        if start > cursor:
            pending.append((cursor, start - cursor))
        cursor = max(cursor, end)
    if cursor < rounded_rows:
        pending.append((cursor, rounded_rows - cursor))
    return pending


def _take_adaptive_range(
    pending_ranges: list[tuple[int, int]],
    controller: AdaptiveBatchSizeController,
) -> tuple[int, int] | None:
    if not pending_ranges:
        return None
    start, remaining = pending_ranges.pop(0)
    size = min(controller.current, remaining)
    if remaining > size:
        pending_ranges.insert(0, (start + size, remaining - size))
    return start, size


def _prepend_grounded_range(pending_ranges: list[tuple[int, int]], start: int, size: int) -> None:
    if size > 0:
        pending_ranges.insert(0, (start, size))


def run_grounded_signal(name: str, cfg: Dict[str, Any], output_dir: Path) -> None:
    """Generate one grounded signal with bounded concurrent, atomically persisted batches."""
    mix_cfg = cfg["mix"][name]
    generation_cfg = cfg.get("generation", {}) or {}
    backend_cfg = cfg.get("backend", {}) or {}
    retry_cfg = backend_cfg.get("retries", {}) or {}
    transient_requeue_delay_seconds = float(retry_cfg.get("exhausted_retryable_requeue_delay_seconds", 60.0))
    batch_size = int(mix_cfg.get("batch_size", generation_cfg.get("batch_size", 32)))
    if not MIN_GROUNDED_BATCH_SIZE <= batch_size <= MAX_GROUNDED_BATCH_SIZE:
        raise ValueError(
            f"Grounded generation supports batch_size between {MIN_GROUNDED_BATCH_SIZE} "
            f"and {MAX_GROUNDED_BATCH_SIZE} for throughput qualification."
        )

    min_batch_size = int(mix_cfg.get("min_batch_size", generation_cfg.get("min_batch_size", 1)))
    if not MIN_GROUNDED_BATCH_SIZE <= min_batch_size <= batch_size:
        raise ValueError("Grounded generation min_batch_size must be between 1 and batch_size")

    token_target, target_rows, rounded_rows = _rounded_batch_target_rows(cfg, mix_cfg, batch_size)
    parallel_requests = int(
        mix_cfg.get(
            "parallel_requests",
            generation_cfg.get("parallel_requests", backend_cfg.get("parallel_requests", 1)),
        )
    )
    if not MIN_GROUNDED_PARALLEL_REQUESTS <= parallel_requests <= MAX_GROUNDED_PARALLEL_REQUESTS:
        raise ValueError(
            "Grounded generation supports parallel_requests between "
            f"{MIN_GROUNDED_PARALLEL_REQUESTS} and {MAX_GROUNDED_PARALLEL_REQUESTS} "
            "for throughput qualification."
        )
    renderer = build_llm(backend_cfg, mix_cfg, role="renderer")
    if renderer.provider != "openrouter":
        raise ValueError("Grounded generation requires backend.provider=openrouter")
    generator = GroundedSignalGenerator(name, renderer, batch_size=batch_size)
    batch_controller = AdaptiveBatchSizeController(
        maximum=batch_size,
        minimum=min_batch_size,
        increase_successes=int(retry_cfg.get("adaptive_batch_size_increase_successes", 4)),
        decrease_factor=float(retry_cfg.get("adaptive_batch_size_decrease_factor", 0.5)),
    )
    store = GroundedBatchStore(output_dir, name)
    reject_writer = JSONLWriter(output_dir / "rejected" / f"{name}.jsonl")

    pending_ranges = _pending_grounded_ranges(rounded_rows, store.terminal_ranges())
    existing_rows = store.materialize_raw()
    print(
        f"[generate] Starting grounded signal: {name} "
        f"(target_tokens_estimate={token_target}, target_rows={target_rows}, "
        f"rounded_rows={rounded_rows}, existing_rows={existing_rows}, "
        f"batch_size={batch_size}, min_batch_size={min_batch_size}, "
        f"parallel_requests={parallel_requests}, model={renderer.model})"
    )
    if not pending_ranges:
        metrics = store.telemetry_summary()
        print(
            f"[generate] Completed grounded signal: {name} rows={existing_rows}, target_rows={rounded_rows}, "
            f"dropped_batches={metrics['dropped_batches']}, dropped_rows={metrics['dropped_rows']}, "
            f"provider_retries={metrics['retryable_provider_retries']}, "
            f"retry_sleep_seconds={metrics['retry_sleep_seconds']:.3f}, "
            f"adaptive_window_increases={metrics['adaptive_window_increases']}, "
            f"adaptive_window_decreases={metrics['adaptive_window_decreases']}, "
            f"adaptive_admission_wait_seconds={metrics['adaptive_admission_wait_seconds']:.3f}, "
            f"adaptive_peak_in_flight_limit={metrics['adaptive_peak_in_flight_limit']}, "
            f"adaptive_min_in_flight_limit={metrics['adaptive_min_in_flight_limit']}, "
            f"max_adaptive_cooldown_seconds={metrics['max_adaptive_cooldown_seconds']:.3f}, "
            f"adaptive_batch_size_observed_minimum={metrics['adaptive_batch_size_observed_minimum']}, "
            f"adaptive_batch_size_observed_peak={metrics['adaptive_batch_size_observed_peak']}, "
            f"adaptive_batch_size_increases={metrics['adaptive_batch_size_increases']}, "
            f"adaptive_batch_size_decreases={metrics['adaptive_batch_size_decreases']}, "
            f"adaptive_batch_size_failures={metrics['adaptive_batch_size_failures']}, "
            f"cost={metrics['cost']:.8f}, request_tokens={metrics['total_tokens']}"
        )
        reject_writer.close()
        return

    active: dict[Any, tuple[int, int]] = {}
    failures: list[tuple[int, int, Exception]] = []
    retryable_requeues: dict[int, int] = {}
    stop_submitting = False

    def submit_available(executor: ThreadPoolExecutor) -> None:
        while not stop_submitting and pending_ranges and len(active) < parallel_requests:
            next_range = _take_adaptive_range(pending_ranges, batch_controller)
            if next_range is None:
                return
            start_index, request_batch_size = next_range
            active[executor.submit(generator.generate_range, start_index, request_batch_size, batch_id=start_index)] = (
                start_index,
                request_batch_size,
            )

    try:
        with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
            submit_available(executor)

            while active:
                done, _ = wait(set(active), return_when=FIRST_COMPLETED)
                for future in done:
                    start_index, request_batch_size = active.pop(future)
                    resubmitted = False
                    try:
                        artifacts, records, telemetry = future.result()
                        batch_controller.record_success()
                        telemetry = {**(telemetry or {}), **batch_controller.snapshot()}
                        store.write_completed(batch_id=start_index, artifacts=artifacts, records=records, telemetry=telemetry)
                        # Batch manifests are the durable checkpoint. Rebuild raw JSONL only at
                        # signal start/resume and completion, not once per completed batch.
                        existing_rows += len(records)
                        usage = telemetry.get("usage", {}) if telemetry else {}
                        cost = float(usage.get("cost", 0.0) or 0.0)
                        print(
                            f"[generate] {name}: batch_start={start_index} batch_size={request_batch_size} "
                            f"rows={existing_rows}/{rounded_rows} cost={cost:.8f}"
                        )
                    except GroundedTransientProviderBatchError as exc:
                        batch_controller.record_failure()
                        retryable_requeues[start_index] = retryable_requeues.get(start_index, 0) + 1
                        reject_writer.write({
                            "signal": name, "architecture": "grounded", "batch_id": start_index,
                            "batch_size": request_batch_size, "status": "requeued_retryable_provider_failure",
                            "requeue_count": retryable_requeues[start_index], "error": str(exc),
                            "telemetry": {**(exc.telemetry or {}), **batch_controller.snapshot()},
                        })
                        print(
                            f"[generate] Requeue transient provider failure: signal={name} batch_start={start_index} "
                            f"batch_size={request_batch_size} next_batch_size={batch_controller.current} "
                            f"requeue_count={retryable_requeues[start_index]} "
                            f"delay={transient_requeue_delay_seconds:.2f}s reason={str(exc)!r}"
                        )
                        retry_size = min(batch_controller.current, request_batch_size)
                        if request_batch_size > retry_size:
                            _prepend_grounded_range(pending_ranges, start_index + retry_size, request_batch_size - retry_size)
                        active[executor.submit(
                            _retry_grounded_batch_after_delay,
                            generator, start_index, retry_size, transient_requeue_delay_seconds
                        )] = (start_index, retry_size)
                        resubmitted = True
                    except GroundedRenderedBatchError as exc:
                        batch_controller.record_failure()
                        telemetry = {**(exc.telemetry or {}), **batch_controller.snapshot()}
                        if request_batch_size > min_batch_size:
                            _prepend_grounded_range(pending_ranges, start_index, request_batch_size)
                            reject_writer.write({
                                "signal": name, "architecture": "grounded", "batch_id": start_index,
                                "batch_size": request_batch_size, "status": "adaptive_batch_size_reduced",
                                "next_batch_size": batch_controller.current,
                                "error": str(exc), "telemetry": telemetry,
                            })
                            print(
                                f"[generate] Reduced adaptive batch size: signal={name} batch_start={start_index} "
                                f"batch_size={request_batch_size} next_batch_size={batch_controller.current} "
                                f"reason={str(exc)!r}"
                            )
                        else:
                            store.write_failed(
                                batch_id=start_index,
                                planned_rows=request_batch_size,
                                artifacts=exc.artifacts,
                                error=exc,
                                telemetry=telemetry,
                                returned_artifact_ids=exc.returned_artifact_ids,
                            )
                            reject_writer.write({
                                "signal": name, "architecture": "grounded", "batch_id": start_index,
                                "batch_size": request_batch_size, "status": "dropped_transient_rendered_failure",
                                "error": str(exc), "telemetry": telemetry,
                            })
                            print(
                                f"[generate] Dropped transient rendered batch: signal={name} batch_start={start_index} "
                                f"planned_rows={request_batch_size} reason={str(exc)!r}"
                            )
                    except Exception as exc:
                        reject_writer.write({
                            "signal": name, "architecture": "grounded", "batch_id": start_index,
                            "batch_size": request_batch_size, "status": "fatal_batch_failure", "error": str(exc),
                        })
                        failures.append((start_index, request_batch_size, exc))
                        stop_submitting = True
                    if not resubmitted:
                        submit_available(executor)
    finally:
        reject_writer.close()

    if failures:
        start_index, _request_batch_size, exc = failures[0]
        raise RuntimeError(
            f"Grounded {name} batch starting at {start_index} failed; rerun to resume from completed batches."
        ) from exc
    final_rows = store.materialize_raw()
    metrics = store.telemetry_summary()
    print(
        f"[generate] Completed grounded signal: {name} rows={final_rows}, "
        f"target_rows={rounded_rows}, batches={metrics['batches']}, "
        f"dropped_batches={metrics['dropped_batches']}, dropped_rows={metrics['dropped_rows']}, "
        f"provider_retries={metrics['retryable_provider_retries']}, "
        f"retry_sleep_seconds={metrics['retry_sleep_seconds']:.3f}, "
        f"adaptive_window_increases={metrics['adaptive_window_increases']}, "
        f"adaptive_window_decreases={metrics['adaptive_window_decreases']}, "
        f"adaptive_admission_wait_seconds={metrics['adaptive_admission_wait_seconds']:.3f}, "
        f"adaptive_peak_in_flight_limit={metrics['adaptive_peak_in_flight_limit']}, "
        f"adaptive_min_in_flight_limit={metrics['adaptive_min_in_flight_limit']}, "
        f"max_adaptive_cooldown_seconds={metrics['max_adaptive_cooldown_seconds']:.3f}, "
        f"adaptive_batch_size_observed_minimum={metrics['adaptive_batch_size_observed_minimum']}, "
        f"adaptive_batch_size_observed_peak={metrics['adaptive_batch_size_observed_peak']}, "
        f"adaptive_batch_size_increases={metrics['adaptive_batch_size_increases']}, "
        f"adaptive_batch_size_decreases={metrics['adaptive_batch_size_decreases']}, "
        f"adaptive_batch_size_failures={metrics['adaptive_batch_size_failures']}, "
        f"cost={metrics['cost']:.8f}, request_tokens={metrics['total_tokens']}"
    )

def run_signal(name: str, cfg: Dict[str, Any], output_dir: Path) -> None:
    mix_cfg = cfg["mix"][name]
    architecture = mix_cfg.get("architecture")
    if architecture != "grounded":
        raise ValueError(
            "Pretrain generation only supports grounded architecture. "
            f"Signal {name!r} has architecture={architecture!r}; set mix.{name}.architecture to 'grounded'."
        )
    run_grounded_signal(name, cfg, output_dir)


def main(config_path: str, signal_override: Optional[str] = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    warn_if_unsupported_model(cfg.get("backend", {}).get("model", ""), context="generate")

    output_dir = _expand_path(cfg["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected").mkdir(parents=True, exist_ok=True)

    if signal_override:
        if signal_override not in cfg["mix"]:
            raise ValueError(f"Unknown signal: {signal_override}")
        run_signal(signal_override, cfg, output_dir)
    else:
        for name in cfg["mix"].keys():
            run_signal(name, cfg, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    args = parser.parse_args()

    main(args.config, signal_override=args.signal)
