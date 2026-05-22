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

load_dotenv()

GENERATOR_MAP = {
    "arithmetic": ArithmeticGenerator,
    "task_code": TaskCodeGenerator,
    "educational_qa_mcq_math": EducationalQAMCQMathGenerator,
    "educational_qa_mcq_general": EducationalQAMCQGeneralGenerator,
    "factual_restraint": FactualRestraintGenerator,
}


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
        retry_sleep_seconds=float(retry_cfg.get("retry_sleep_seconds", 0.5)),
        retry_backoff_initial_seconds=float(retry_cfg.get("retry_backoff_initial_seconds", 1.0)),
        retry_backoff_max_seconds=float(retry_cfg.get("retry_backoff_max_seconds", 30.0)),
        retry_backoff_multiplier=float(retry_cfg.get("retry_backoff_multiplier", 2.0)),
        retry_jitter_ratio=float(retry_cfg.get("retry_jitter_ratio", 0.30)),
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

def run_signal(name: str, cfg: Dict[str, Any], output_dir: Path) -> None:
    mix_cfg = cfg["mix"][name]
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
