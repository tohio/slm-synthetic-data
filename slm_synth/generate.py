#!/usr/bin/env python3
import argparse
import yaml
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from slm_synth.llm import LLMBackend
from slm_synth.writer import JSONLWriter
from slm_synth.prompt_loader import load_prompt
from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.task_code import TaskCodeGenerator
from slm_synth.sources.educational_qa_mcq import EducationalQAMCQGenerator
from slm_synth.sources.factual_restraint import FactualRestraintGenerator

GENERATOR_MAP = {
    "arithmetic": ArithmeticGenerator,
    "task_code": TaskCodeGenerator,
    "educational_qa_mcq": EducationalQAMCQGenerator,
    "factual_restraint": FactualRestraintGenerator,
}

def run_signal(name, cfg, llm, output_dir, batch_size):
    share = cfg["mix"][name]["share"]
    prompt_file = cfg["mix"][name]["prompt_file"]
    module = cfg["mix"][name]["source_module"]

    total_tokens = cfg["target_total_tokens"]
    samples = int(total_tokens * share / 10)  # ~10 tokens per sample avg

    print(f"[generate] Starting signal: {name} ({samples} samples)")

    raw_path = output_dir / "raw" / f"{name}.jsonl"
    rejected_path = output_dir / "rejected" / f"{name}.jsonl"

    writer = JSONLWriter(raw_path)
    reject_writer = JSONLWriter(rejected_path)

    GenClass = GENERATOR_MAP[name]
    generator = GenClass(llm, prompt_file, batch_size=batch_size)

    generated = 0
    failures = 0
    max_failures = cfg["validation"]["max_retries"]

    while generated < samples:
        try:
            batch = generator.generate_batch()

            for obj in batch:
                if generated >= samples:
                    break
                writer.write(obj)
                generated += 1

            if generated % 100 == 0:
                print(f"[generate] {name}: {generated}/{samples}")

        except Exception as e:
            failures += 1
            reject_writer.write({"error": str(e)})
            print(f"[generate] ERROR in {name}: {e}")

            if failures >= max_failures:
                print(f"[generate] FATAL: Too many failures in '{name}'. Aborting.")
                break

            time.sleep(0.2)

    writer.close()
    reject_writer.close()

    print(f"[generate] Completed signal: {name}")

def main(config_path, signal_override=None):
    cfg = yaml.safe_load(Path(config_path).read_text())

    gen_cfg = cfg.get("generation", {})
    batch_size = int(gen_cfg.get("batch_size", 1))

    output_dir = Path(cfg["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected").mkdir(parents=True, exist_ok=True)

    llm = LLMBackend(
        provider=cfg["backend"]["provider"],
        model=cfg["backend"]["model"],
        max_tokens=cfg["backend"]["max_tokens"],
        temperature=cfg["backend"]["temperature"],
        top_p=cfg["backend"]["top_p"],
    )

    if signal_override:
        run_signal(signal_override, cfg, llm, output_dir, batch_size)
    else:
        for name in cfg["mix"].keys():
            run_signal(name, cfg, llm, output_dir, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    args = parser.parse_args()

    main(args.config, signal_override=args.signal)
