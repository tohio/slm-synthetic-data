import os
import json
from pathlib import Path
import yaml
import traceback
import sys
import time

from slm_synth.llm import LLMBackend
from slm_synth.rate_limit import RateLimiter

from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.task_code import TaskCodeGenerator
from slm_synth.sources.educational_qa_mcq import EducationalQAMCQGenerator
from slm_synth.sources.factual_restraint import FactualRestraintGenerator


def log(msg: str):
    print(f"[generate] {msg}", flush=True)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


GENERATOR_MAP = {
    "arithmetic": ArithmeticGenerator,
    "task_code": TaskCodeGenerator,
    "educational_qa_mcq": EducationalQAMCQGenerator,
    "factual_restraint": FactualRestraintGenerator,
}


# ------------------------------------------------------------
# Run ONE signal family (batched)
# ------------------------------------------------------------
def run_one_signal(name: str, cfg, llm, output_dir: Path):
    log(f"Starting signal: {name}")

    spec = cfg["mix"][name]
    share = spec["share"]

    total_tokens = cfg["target_total_tokens"]
    target_tokens = int(total_tokens * share)

    batch_size = cfg.get("generation", {}).get("batch_size", 8)

    generator_cls = GENERATOR_MAP[name]
    generator = generator_cls(llm, spec["prompt_file"])

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_file = raw_dir / f"{name}.jsonl"

    rate = RateLimiter(cfg)

    tokens_generated = 0
    attempt = 0

    with open(out_file, "w", encoding="utf-8") as f:
        while tokens_generated < target_tokens:
            try:
                batch = generator.generate_batch(batch_size)

                for obj in batch:
                    text = json.dumps(obj, ensure_ascii=False)
                    f.write(text + "\n")
                    tokens_generated += len(text.split())

                    if tokens_generated >= target_tokens:
                        break

                log(f"{name}: {tokens_generated}/{target_tokens} tokens")

                attempt = 0
                rate.sleep_with_jitter()

            except Exception as e:
                log(f"ERROR in signal '{name}': {e}")
                traceback.print_exc()
                attempt += 1

                if attempt > 5:
                    log(f"FATAL: Too many failures in '{name}'. Aborting.")
                    raise

                rate.backoff(attempt)
                continue

    log(f"Completed signal: {name}")


# ------------------------------------------------------------
# Run ALL signals sequentially
# ------------------------------------------------------------
def run_all_signals(cfg, llm, output_dir: Path):
    for name in cfg["mix"].keys():
        run_one_signal(name, cfg, llm, output_dir)


# ------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------
def main(config_path: str, signal_override: str | None = None):
    log(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    # Resolve output directory
    output_dir_raw = cfg["output_dir"]
    output_dir = Path(os.path.expandvars(output_dir_raw))
    log(f"Resolved output_dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    backend_cfg = cfg["backend"]
    log(f"Backend provider: {backend_cfg['provider']}")
    log(f"Backend model: {backend_cfg['model']}")
    log(f"Max tokens per call: {backend_cfg['max_tokens']}")
    log(f"Temperature: {backend_cfg['temperature']}")
    log(f"Top-p: {backend_cfg['top_p']}")

    llm = LLMBackend(
        provider=backend_cfg["provider"],
        model=backend_cfg["model"],
        max_tokens=backend_cfg["max_tokens"],
        temperature=backend_cfg["temperature"],
        top_p=backend_cfg["top_p"],
    )

    log(f"Target total tokens: {cfg['target_total_tokens']}")

    if signal_override:
        if signal_override not in cfg["mix"]:
            raise ValueError(f"Unknown signal: {signal_override}")
        log(f"Running only signal: {signal_override}")
        run_one_signal(signal_override, cfg, llm, output_dir)
    else:
        log("Running all signals...")
        run_all_signals(cfg, llm, output_dir)

    log("Generation complete.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to synthetic.yaml")
    parser.add_argument("--signal", required=False, help="Run only one signal family")
    args = parser.parse_args()

    main(args.config, signal_override=args.signal)
