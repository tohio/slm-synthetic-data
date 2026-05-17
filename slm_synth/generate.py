import os
import json
from pathlib import Path
from slm_synth.llm import LLMBackend
from slm_synth.rate_limit import RateLimiter
from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.task_code import TaskCodeGenerator
from slm_synth.sources.educational_qa_mcq import EducationalQAMCQGenerator
from slm_synth.sources.factual_restraint import FactualRestraintGenerator
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


GENERATOR_MAP = {
    "arithmetic": ArithmeticGenerator,
    "task_code": TaskCodeGenerator,
    "educational_qa_mcq": EducationalQAMCQGenerator,
    "factual_restraint": FactualRestraintGenerator,
}


def main(config_path: str):
    cfg = load_config(config_path)

    output_dir = Path(cfg["output_dir"])
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # LLM backend
    backend_cfg = cfg["backend"]
    llm = LLMBackend(
        provider=backend_cfg["provider"],
        model=backend_cfg["model"],
        max_tokens=backend_cfg["max_tokens"],
        temperature=backend_cfg["temperature"],
        top_p=backend_cfg["top_p"],
    )

    # Compute per-signal token budgets
    total_tokens = cfg["target_total_tokens"]

    for name, spec in cfg["mix"].items():
        share = spec["share"]
        target_tokens = int(total_tokens * share)

        generator_cls = GENERATOR_MAP[name]
        generator = generator_cls(llm, spec["prompt_file"])

        out_file = raw_dir / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            tokens_generated = 0

            rate = RateLimiter(cfg) 

            while tokens_generated < target_tokens:
                try:
                    obj = generator.generate()
                    text = json.dumps(obj, ensure_ascii=False)
                    f.write(text + "\n")
                    tokens_generated += len(text.split())

                    rate.sleep_with_jitter()

                except Exception:

                    rate.backoff(attempt=0)

                    continue


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
