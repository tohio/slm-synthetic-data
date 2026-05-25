#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv

from slm_synth.diversity import build_diversity_context
from slm_synth.llm import LLMBackend
from slm_synth.model_support import warn_if_unsupported_model
from slm_synth.rate_limit import RateLimiter
from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.educational_qa_mcq_general import EducationalQAMCQGeneralGenerator
from slm_synth.sources.educational_qa_mcq_math import EducationalQAMCQMathGenerator
from slm_synth.sources.factual_restraint import FactualRestraintGenerator
from slm_synth.sources.task_code import TaskCodeGenerator
from slm_synth.writer import JSONLWriter
from slm_synth.grounded import GroundedBatchStore, GroundedRenderedBatchError, GroundedSignalGenerator

load_dotenv()

GENERATOR_MAP = {
    "arithmetic": ArithmeticGenerator,
    "task_code": TaskCodeGenerator,
    "educational_qa_mcq_math": EducationalQAMCQMathGenerator,
    "educational_qa_mcq_general": EducationalQAMCQGeneralGenerator,
    "factual_restraint": FactualRestraintGenerator,
}

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
        provider=base_cfg.get("provider", "groq"),
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
        shared_throttle_enabled=bool(retry_cfg.get("shared_throttle_enabled", True)),
        shared_throttle_burst_threshold=int(retry_cfg.get("shared_throttle_burst_threshold", 8)),
        shared_throttle_window_seconds=float(retry_cfg.get("shared_throttle_window_seconds", 2.0)),
        shared_throttle_initial_cooldown_seconds=float(retry_cfg.get("shared_throttle_initial_cooldown_seconds", 5.0)),
        shared_throttle_max_cooldown_seconds=float(retry_cfg.get("shared_throttle_max_cooldown_seconds", 120.0)),
        shared_throttle_multiplier=float(retry_cfg.get("shared_throttle_multiplier", 2.0)),
        shared_throttle_success_reset_count=int(retry_cfg.get("shared_throttle_success_reset_count", 32)),
        require_parameters=bool(base_cfg.get("require_parameters", True)),
        allow_fallbacks=bool(base_cfg.get("allow_fallbacks", False)),
    )


def signal_sample_target(name: str, cfg: Dict[str, Any], signal_cfg: Dict[str, Any]) -> int:
    generation_cfg = cfg.get("generation", {}) or {}

    explicit_signal_samples = signal_cfg.get("samples")
    if explicit_signal_samples is not None:
        return int(explicit_signal_samples)

    # Backward compatible escape hatch for tiny tests.
    samples_per_signal = generation_cfg.get("samples_per_signal")
    if samples_per_signal is not None and int(samples_per_signal) > 0:
        share = float(signal_cfg.get("share", 1.0))
        return max(1, int(int(samples_per_signal) * share))

    target_tokens = int(cfg["target_total_tokens"])
    share = float(signal_cfg.get("share", 0.0))
    avg_tokens = int(signal_cfg.get("avg_tokens_per_sample", generation_cfg.get("avg_tokens_per_sample", 80)))
    return max(1, int(target_tokens * share / avg_tokens))


def generate_with_split(generator: Any, batch_size: int, min_batch_size: int) -> List[Dict[str, Any]]:
    """
    Generate a batch. If a larger batch fails after request-level retries,
    split it recursively rather than aborting the whole signal.
    """
    original_batch_size = generator.batch_size
    try:
        generator.batch_size = batch_size
        return generator.generate_batch()
    except Exception:
        if batch_size <= min_batch_size:
            raise
        left = max(min_batch_size, batch_size // 2)
        right = batch_size - left
        if right <= 0:
            raise
        return generate_with_split(generator, left, min_batch_size) + generate_with_split(
            generator, right, min_batch_size
        )
    finally:
        generator.batch_size = original_batch_size


def submit_next(
    executor: ThreadPoolExecutor,
    GenClass: Any,
    candidate_llm: LLMBackend,
    response_llm: LLMBackend,
    prompt_file: Optional[str],
    batch_size: int,
    min_batch_size: int,
    signal_name: str,
    batch_id: int,
    diversity_enabled: bool,
):
    diversity_context = build_diversity_context(signal_name, batch_id) if diversity_enabled else ""
    generator = GenClass(
        candidate_llm,
        response_llm=response_llm,
        prompt_file=prompt_file,
        batch_size=batch_size,
        diversity_context=diversity_context,
    )
    return executor.submit(generate_with_split, generator, batch_size, min_batch_size)




def _submit_delay(rate_limiter: RateLimiter) -> None:
    # Small launch pacing prevents synchronized bursts across worker threads.
    # Request-level backoff in LLMBackend handles 429/498/5xx after submission.
    rate_limiter.sleep_with_jitter()


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


def run_grounded_signal(name: str, cfg: Dict[str, Any], output_dir: Path) -> None:
    """Generate one grounded signal with bounded concurrent, atomically persisted batches."""
    mix_cfg = cfg["mix"][name]
    generation_cfg = cfg.get("generation", {}) or {}
    backend_cfg = cfg.get("backend", {}) or {}
    batch_size = int(mix_cfg.get("batch_size", generation_cfg.get("batch_size", 32)))
    if not MIN_GROUNDED_BATCH_SIZE <= batch_size <= MAX_GROUNDED_BATCH_SIZE:
        raise ValueError(
            f"Grounded generation supports batch_size between {MIN_GROUNDED_BATCH_SIZE} "
            f"and {MAX_GROUNDED_BATCH_SIZE} for throughput qualification."
        )

    token_target, target_rows, rounded_rows = _rounded_batch_target_rows(cfg, mix_cfg, batch_size)
    total_batches = rounded_rows // batch_size
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
    store = GroundedBatchStore(output_dir, name)
    reject_writer = JSONLWriter(output_dir / "rejected" / f"{name}.jsonl")

    terminal = set(store.terminal_batch_ids())
    pending_ids = [batch_id for batch_id in range(total_batches) if batch_id not in terminal]
    existing_rows = store.materialize_raw()
    print(
        f"[generate] Starting grounded signal: {name} "
        f"(target_tokens_estimate={token_target}, target_rows={target_rows}, "
        f"rounded_rows={rounded_rows}, existing_rows={existing_rows}, "
        f"batch_size={batch_size}, parallel_requests={parallel_requests}, model={renderer.model})"
    )
    if not pending_ids:
        metrics = store.telemetry_summary()
        print(
            f"[generate] Completed grounded signal: {name} rows={existing_rows}, target_rows={rounded_rows}, "
            f"dropped_batches={metrics['dropped_batches']}, dropped_rows={metrics['dropped_rows']}, "
            f"provider_retries={metrics['retryable_provider_retries']}, "
            f"retry_sleep_seconds={metrics['retry_sleep_seconds']:.3f}, "
            f"shared_throttle_trips={metrics['shared_throttle_trips']}, "
            f"shared_throttle_wait_seconds={metrics['shared_throttle_wait_seconds']:.3f}, "
            f"max_shared_throttle_cooldown_seconds={metrics['max_shared_throttle_cooldown_seconds']:.3f}, "
            f"cost={metrics['cost']:.8f}, request_tokens={metrics['total_tokens']}"
        )
        reject_writer.close()
        return

    active: dict[Any, int] = {}
    next_position = 0
    failures: list[tuple[int, Exception]] = []
    stop_submitting = False
    try:
        with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
            while next_position < len(pending_ids) and len(active) < parallel_requests:
                batch_id = pending_ids[next_position]
                active[executor.submit(generator.generate_batch, batch_id)] = batch_id
                next_position += 1

            while active:
                done, _ = wait(set(active), return_when=FIRST_COMPLETED)
                for future in done:
                    batch_id = active.pop(future)
                    try:
                        artifacts, records, telemetry = future.result()
                        store.write_completed(batch_id=batch_id, artifacts=artifacts, records=records, telemetry=telemetry)
                        existing_rows = store.materialize_raw()
                        usage = telemetry.get("usage", {}) if telemetry else {}
                        cost = float(usage.get("cost", 0.0) or 0.0)
                        print(f"[generate] {name}: batch={batch_id} rows={existing_rows}/{rounded_rows} cost={cost:.8f}")
                    except GroundedRenderedBatchError as exc:
                        store.write_failed(
                            batch_id=batch_id,
                            planned_rows=batch_size,
                            artifacts=exc.artifacts,
                            error=exc,
                            telemetry=exc.telemetry,
                            returned_artifact_ids=exc.returned_artifact_ids,
                        )
                        reject_writer.write({
                            "signal": name, "architecture": "grounded", "batch_id": batch_id,
                            "batch_size": batch_size, "status": "dropped_transient_rendered_failure",
                            "error": str(exc),
                        })
                        print(
                            f"[generate] Dropped transient rendered batch: signal={name} batch={batch_id} "
                            f"planned_rows={batch_size} reason={str(exc)!r}"
                        )
                    except Exception as exc:
                        reject_writer.write({
                            "signal": name, "architecture": "grounded", "batch_id": batch_id,
                            "batch_size": batch_size, "status": "fatal_batch_failure", "error": str(exc),
                        })
                        failures.append((batch_id, exc))
                        stop_submitting = True
                    if not stop_submitting and next_position < len(pending_ids):
                        next_id = pending_ids[next_position]
                        active[executor.submit(generator.generate_batch, next_id)] = next_id
                        next_position += 1
    finally:
        reject_writer.close()

    if failures:
        batch_id, exc = failures[0]
        raise RuntimeError(
            f"Grounded {name} batch {batch_id} failed; rerun to resume from completed batches."
        ) from exc
    metrics = store.telemetry_summary()
    print(
        f"[generate] Completed grounded signal: {name} rows={store.materialize_raw()}, "
        f"target_rows={rounded_rows}, batches={metrics['batches']}, "
        f"dropped_batches={metrics['dropped_batches']}, dropped_rows={metrics['dropped_rows']}, "
        f"provider_retries={metrics['retryable_provider_retries']}, "
        f"retry_sleep_seconds={metrics['retry_sleep_seconds']:.3f}, "
        f"shared_throttle_trips={metrics['shared_throttle_trips']}, "
        f"shared_throttle_wait_seconds={metrics['shared_throttle_wait_seconds']:.3f}, "
        f"max_shared_throttle_cooldown_seconds={metrics['max_shared_throttle_cooldown_seconds']:.3f}, "
        f"cost={metrics['cost']:.8f}, request_tokens={metrics['total_tokens']}"
    )

def run_signal(name: str, cfg: Dict[str, Any], output_dir: Path) -> None:
    mix_cfg = cfg["mix"][name]
    if mix_cfg.get("architecture") == "grounded":
        run_grounded_signal(name, cfg, output_dir)
        return
    generation_cfg = cfg.get("generation", {}) or {}
    backend_cfg = cfg.get("backend", {}) or {}
    rate_cfg = cfg.get("rate_limit", {}) or {}
    rate_limiter = RateLimiter(cfg)

    batch_size = int(mix_cfg.get("batch_size", generation_cfg.get("batch_size", 1)))
    min_batch_size = int(mix_cfg.get("min_batch_size", generation_cfg.get("min_batch_size", 1)))
    parallel_requests = int(
        mix_cfg.get(
            "parallel_requests",
            backend_cfg.get("parallel_requests", rate_cfg.get("max_concurrency", 1)),
        )
    )
    max_rejected_batches = int(
        mix_cfg.get("max_rejected_batches", generation_cfg.get("max_rejected_batches", 1000))
    )
    prompt_file = mix_cfg.get("prompt_file")
    diversity_cfg = generation_cfg.get("diversity", {}) or {}
    diversity_enabled = bool(mix_cfg.get("diversity_enabled", diversity_cfg.get("enabled", True)))
    samples = signal_sample_target(name, cfg, mix_cfg)

    candidate_llm = build_llm(backend_cfg, mix_cfg, role="candidate")
    response_llm = build_llm(backend_cfg, mix_cfg, role="response")

    print(
        f"[generate] Starting signal: {name} "
        f"({samples} samples, batch_size={batch_size}, parallel_requests={parallel_requests}, diversity={diversity_enabled}, "
        f"candidate_model={candidate_llm.model}, response_model={response_llm.model})"
    )

    raw_path = output_dir / "raw" / f"{name}.jsonl"
    rejected_path = output_dir / "rejected" / f"{name}.jsonl"

    writer = JSONLWriter(raw_path)
    reject_writer = JSONLWriter(rejected_path)

    GenClass = GENERATOR_MAP[name]

    generated = 0
    rejected_batches = 0
    submitted = 0
    pending = set()

    try:
        with ThreadPoolExecutor(max_workers=max(1, parallel_requests)) as executor:
            while len(pending) < parallel_requests and submitted * batch_size < samples:
                _submit_delay(rate_limiter)
                pending.add(submit_next(executor, GenClass, candidate_llm, response_llm, prompt_file, batch_size, min_batch_size, name, submitted, diversity_enabled))
                submitted += 1

            last_log = time.time()
            while pending and generated < samples:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        batch = future.result()
                        for obj in batch:
                            if generated >= samples:
                                break
                            writer.write(obj)
                            generated += 1
                    except Exception as exc:
                        rejected_batches += 1
                        reject_writer.write(
                            {
                                "signal": name,
                                "batch_size": batch_size,
                                "error": str(exc),
                            }
                        )
                        print(f"[generate] ERROR in {name}: {exc}")
                        if rejected_batches >= max_rejected_batches:
                            raise RuntimeError(
                                f"Too many rejected batches in '{name}' ({rejected_batches})."
                            ) from exc

                    while len(pending) < parallel_requests and generated + len(pending) * batch_size < samples:
                        _submit_delay(rate_limiter)
                        pending.add(
                            submit_next(executor, GenClass, candidate_llm, response_llm, prompt_file, batch_size, min_batch_size, name, submitted, diversity_enabled)
                        )
                        submitted += 1

                now = time.time()
                if generated % 100 == 0 or now - last_log >= 10:
                    print(
                        f"[generate] {name}: {generated}/{samples} "
                        f"accepted, rejected_batches={rejected_batches}"
                    )
                    last_log = now
    finally:
        writer.close()
        reject_writer.close()

    print(
        f"[generate] Completed signal: {name} "
        f"accepted={generated}, rejected_batches={rejected_batches}"
    )


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
