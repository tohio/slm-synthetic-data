# SLM Synthetic Data

Synthetic data generation for the SLM projects.

This repository has two separate workflows:

| Workflow | Package | Output | Purpose |
|---|---|---|---|
| Pretraining synthetic data | `slm_synth.pretrain` | Validated and deduped JSONL records | Targeted signals for pretraining or continued pretraining |
| Response distillation data | `slm_synth.distillation` | One JSONL dataset per signal | Prompt-response datasets for teacher response distillation |

OpenRouter is the only supported production provider. Shared provider, retry, concurrency, and structured-output logic lives in `slm_synth/llm.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...       # only needed for publishing
```

## Pretraining Workflow

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
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped
```

Production-sized configuration:

```bash
make production-config CONCURRENCY=4
```

## Distillation Workflow

The distillation workflow builds local prompt records, asks a teacher for `reasoning` and `response`, validates teacher ids, and writes public rows with this schema:

```json
{"id": "string", "prompt": "string", "reasoning": null, "response": "string"}
```

`reasoning` may also be a list of strings. Public rows do not include `signal`, `metadata`, `teacher_model`, `teacher_provider`, `generation_run`, or `difficulty`; those fields stay in local manifests and dataset cards.

Supported distillation signals:

| Signal |
|---|
| `arithmetic` |
| `code` |
| `debugging` |
| `database` |
| `cloud` |
| `data_transform` |
| `educational_qa` |
| `factual_restraint` |
| `planning` |
| `instruction` |

Plan a token target:

```bash
make distill-plan DISTILL_TARGET=smoke
```

Generate a seed run across all distillation signals:

```bash
make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=smoke-001 \
  DISTILL_TARGET=smoke
```

Build a dataset card from the run manifest:

```bash
make distill-build-dataset-card \
  DISTILL_RUN_MANIFEST=data/distillation/manifests/smoke-001.manifest.json \
  DISTILL_DATASET_CARD=data/distillation/README.md \
  DISTILL_DATASET_NAME="SLM Synthetic Distillation Smoke"
```

Token target presets:

| Preset | Total tokens |
|---|---:|
| `smoke` | 100K |
| `pilot` | 1M |
| `scale-check` | 10M |
| `final` | 100M |

## Repository Layout

```text
configs/                       Pretraining config generator and template
prompts/                       Pretraining prompt helpers
slm_synth/llm.py               Shared OpenRouter client, retries, and concurrency
slm_synth/pretrain/            Pretraining generation, validation, dedup, reports, publish
slm_synth/distillation/        Response distillation schema, prompts, generation, manifests
tests/                         Unit tests for pretraining and distillation workflows
docs/                          Command reference and supporting project docs
```

Compatibility wrappers remain at older `slm_synth.*` module paths while the pretraining implementation lives under `slm_synth.pretrain`.

Full command reference: [docs/COMMANDS.md](docs/COMMANDS.md)
