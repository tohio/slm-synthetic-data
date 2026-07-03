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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pip install pytest
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...       # only needed for publishing
```

## Pretraining

The pretraining workflow generates grounded synthetic records, validates them, exact-deduplicates them, and can publish the final corpus.

Supported pretraining signals:

| Signal | Stored record |
|---|---|
| `arithmetic` | `question`, `steps`, `answer` |
| `task_code` | `task`, `plan`, `code` |
| `educational_qa_mcq_math` | `question`, `choices`, `correct_index`, `explanation` |
| `educational_qa_mcq_general` | `evidence`, `question`, `choices`, `correct_index`, `explanation` |
| `factual_restraint` | `question`, `safe_answer` |

Smoke run:

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

## Taxonomy And Holdouts

| Field | Meaning |
|---|---|
| `category` | Training objective |
| `eval_family` | Eval-shaped behavior pattern |
| `template_family` | Generation/template surface |
| `failure_mode` | DPO-only rejected-answer behavior |
| `holdout_key` | Exact local structured holdout guard |

Training can cover the same task family as an eval with different variables or templates. It must not include exact eval prompts or structured holdout-key matches.

## Validation

```bash
python -m compileall -q slm_synth tests
pytest -q
```

Focused SFT/DPO/distillation checks:

```bash
pytest -q \
  tests/test_sft_*.py \
  tests/test_dpo_*.py \
  tests/test_distillation_*.py \
  tests/test_taxonomy.py \
  tests/test_eval_holdouts.py \
  tests/test_pretrain_manifest.py
```

## Repository Layout

```text
configs/                       Pretraining config generator and template
prompts/                       Pretraining prompt helpers
slm_synth/llm.py               Shared OpenRouter client, retries, and concurrency
slm_synth/pretrain/            Pretraining generation, validation, dedup, reports, publish
slm_synth/distillation/        Response distillation schema, prompts, generation, manifests
slm_synth/sft/                 SFT schema, task specs, seed data, LLM batches, manifests
slm_synth/dpo/                 DPO schema, task specs, seed data, LLM batches, manifests
slm_synth/taxonomy/            Taxonomy labels and holdout registry
tests/                         Unit tests for all workflows
docs/                          Command reference and supporting project docs
```

Compatibility wrappers remain at older `slm_synth.*` module paths while the pretraining implementation lives under `slm_synth.pretrain`.

Full command reference: [docs/COMMANDS.md](docs/COMMANDS.md)
