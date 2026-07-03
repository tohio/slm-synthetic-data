# SLM Synthetic Data

Synthetic data generation for the SLM projects.

The repository has four workflows:

| Workflow | Package | Public output | Purpose |
|---|---|---|---|
| Pretraining synthetic data | `slm_synth.pretrain` | Validated and deduped JSONL records | Targeted signals for pretraining or continued pretraining |
| Response distillation | `slm_synth.distillation` | Prompt-response JSONL rows | Teacher response distillation datasets |
| SFT | `slm_synth.sft` | Chat-message JSONL rows | Supervised instruction/chat examples |
| DPO | `slm_synth.dpo` | Preference JSONL rows | Chosen/rejected preference examples |

OpenRouter is the only supported production provider. Shared provider, retry, concurrency, and structured-output logic lives in `slm_synth/llm.py`. Unsupported providers fail fast.

## Setup

```bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...       # only needed for Hugging Face publishing
```

## Choose A Generation Target

Pick the dataset you want to produce, then run the matching section below.

| Target | Use This Section | Main Output |
|---|---|---|
| Pretraining signals | [Generate Pretraining Data](#generate-pretraining-data) | `data/runs/<run>/deduped` |
| Teacher response data | [Generate Response Distillation Data](#generate-response-distillation-data) | `data/distillation/datasets` |
| Supervised chat data | [Generate SFT Data](#generate-sft-data) | `data/sft/datasets` |
| Preference data | [Generate DPO Data](#generate-dpo-data) | `data/dpo/datasets` |

For a new run, start with the small command first, inspect the output, then run the target command with the row or token count you want.

## Generate Pretraining Data

Pretraining data is generated from configured grounded signals, then validated and deduped.

Supported signals:

```text
arithmetic
task_code
educational_qa_mcq_math
educational_qa_mcq_general
factual_restraint
```

### Small Smoke Run

```bash
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_smoke
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup
make pretrain-generate-manifest
make pretrain-report-coverage
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped
```

Production-sized configuration:

```bash
make production-config CONCURRENCY=4
```

## Response Distillation

Distillation rows use this public schema:

```json
{"id": "string", "prompt": "string", "reasoning": null, "response": "string"}
```

`reasoning` may also be a list of strings. Teacher model, provider, generation run, token target, signal, and internal metadata stay in manifests and dataset cards.

Generate a seed run:

```bash
make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=distill-smoke-001 \
  DISTILL_TARGET=smoke
```

Build a dataset card:

```bash
make distill-build-dataset-card \
  DISTILL_RUN_MANIFEST=data/distillation/manifests/distill-smoke-001.manifest.json \
  DISTILL_DATASET_CARD=data/distillation/README.md \
  DISTILL_DATASET_NAME="SLM Synthetic Distillation Smoke"
```

## SFT

SFT rows use this public schema:

```json
{
  "id": "string",
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null"
  }
}
```

Build no-network task specs:

```bash
make sft-build-specs \
  SFT_SPEC_FAMILY=basic_arithmetic_qa \
  SFT_COUNT=2 \
  SFT_SPECS=/tmp/sft.specs.jsonl
```

Generate a live OpenRouter run:

```bash
make sft-generate-llm-run \
  SFT_FAMILIES=basic_arithmetic_qa \
  SFT_COUNT_PER_FAMILY=2 \
  SFT_BATCH_SIZE=2 \
  SFT_MAX_WORKERS=1 \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-smoke-001 \
  SFT_MAX_TOKENS=2048
```

## DPO

DPO rows use this public schema:

```json
{
  "id": "string",
  "prompt": [{"role": "user", "content": "string"}],
  "chosen": [{"role": "assistant", "content": "string"}],
  "rejected": [{"role": "assistant", "content": "string"}],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null",
    "failure_mode": "string"
  }
}
```

Build no-network task specs:

```bash
make dpo-build-specs \
  DPO_SPEC_FAMILY=basic_arithmetic_qa \
  DPO_COUNT=2 \
  DPO_SPECS=/tmp/dpo.specs.jsonl
```

Generate a live OpenRouter run:

```bash
make dpo-generate-llm-run \
  DPO_FAMILIES=basic_arithmetic_qa \
  DPO_COUNT_PER_FAMILY=2 \
  DPO_BATCH_SIZE=2 \
  DPO_MAX_WORKERS=1 \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-smoke-001 \
  DPO_MAX_TOKENS=2048
```

### Generate Target DPO Data

Set `DPO_COUNT_PER_FAMILY` to the target number of rows per family.

```bash
make dpo-generate-llm-run \
  DPO_FAMILIES=all \
  DPO_COUNT_PER_FAMILY=1000 \
  DPO_BATCH_SIZE=5 \
  DPO_MAX_WORKERS=2 \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-target-001 \
  DPO_MAX_TOKENS=4096

make dpo-report-coverage
```

For a conservative CPU node, keep `DPO_MAX_WORKERS=1`. Increase to `2` only after a clean smoke run.

## Generate Multiple Dataset Types

Use this order when you want a small end-to-end data generation pass across the repo:

```bash
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_smoke
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup

make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=distill-smoke-001 \
  DISTILL_TARGET=smoke

make sft-generate-llm-run \
  SFT_FAMILIES=basic_arithmetic_qa \
  SFT_COUNT_PER_FAMILY=2 \
  SFT_BATCH_SIZE=2 \
  SFT_MAX_WORKERS=1 \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-smoke-001 \
  SFT_MAX_TOKENS=2048

make dpo-generate-llm-run \
  DPO_FAMILIES=basic_arithmetic_qa \
  DPO_COUNT_PER_FAMILY=2 \
  DPO_BATCH_SIZE=2 \
  DPO_MAX_WORKERS=1 \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-smoke-001 \
  DPO_MAX_TOKENS=2048
```

After those outputs look correct, run the target commands for the dataset families you need.

## Output Locations

| Dataset | Public Data | Manifests And Reports |
|---|---|---|
| Pretraining | `data/runs/<run>/deduped` | `data/runs/<run>/...` |
| Distillation | `data/distillation/datasets` | `data/distillation/manifests`, `data/distillation/coverage.json` |
| SFT | `data/sft/datasets` | `data/sft/manifests`, `data/sft/coverage.json` |
| DPO | `data/dpo/datasets` | `data/dpo/manifests`, `data/dpo/coverage.json` |

## Holdout Policy

Training examples may use the same task family as evals with different variables or templates. They must not include exact eval prompts or matching structured `holdout_key` values.

Taxonomy fields:

| Field | Meaning |
|---|---|
| `category` | Training objective |
| `eval_family` | Eval-shaped behavior pattern |
| `template_family` | Generation/template surface |
| `failure_mode` | DPO-only rejected-answer behavior |
| `holdout_key` | Exact local structured holdout guard |

## Optional Test Commands

Tests are for repo maintenance and pre-scale confidence checks. They are not required as the first user step.

Run the full suite:

```bash
python -m compileall -q slm_synth tests
pytest -q
```

Focused validation:

```bash
pytest -q \
  tests/test_sft_*.py \
  tests/test_dpo_*.py \
  tests/test_distillation_*.py \
  tests/test_taxonomy.py \
  tests/test_eval_holdouts.py \
  tests/test_pretrain_manifest.py
```

## Command Reference

See [docs/COMMANDS.md](docs/COMMANDS.md) for every Make variable and command.
