# SLM Synthetic Data

Synthetic dataset generation and quality control for the SLM training stack.

## Overview

`slm-synthetic-data` builds five production dataset products: pretraining, generic SFT, generic DPO, distillation SFT, and distillation DPO. Each product has one supported generation pipeline, one quality path, one reporting path, and one Hugging Face publication path.

The repository is intentionally limited to **dataset creation**. Model training, student sampling, logits, evaluation, checkpoints, and model export belong in downstream repositories.

## Architecture

All five datasets share mechanical execution primitives under `slm_synth/runtime/`, while each dataset package owns its own semantics.

~~~text
                         ┌───────────────────────┐
                         │  slm_synth/runtime/   │
                         │ backend · batching    │
                         │ stages · novelty · IO │
                         └───────────┬───────────┘
                                     │
          ┌──────────────┬───────────┼───────────┬────────────────┐
          │              │           │           │                │
      pretrain/        sft/         dpo/   distillation_sft/ distillation_dpo/
          │              │           │           │                │
          └──────────────┴───────────┴───────────┴────────────────┘
                                     │
                     validated public datasets + manifests
                                     │
                              Hugging Face
~~~

The runtime owns provider calls, strict structured output, batching, concurrency, retry/isolation, novelty mechanics, JSON/JSONL IO, and shared reporting helpers. Dataset packages own prompts, schemas, deterministic checks, judge/reviewer criteria, acceptance rules, metadata, reports, and publication layout.

See [Architecture](docs/ARCHITECTURE.md) for the detailed component and data-flow model.

## Features

- **Five explicit dataset products** with no hidden legacy generation path.
- **Strict structured generation** through OpenRouter-compatible JSON Schema requests.
- **Configurable model roles** for derivation, generation, judging, and review.
- **Provider fallback and routing controls** for throttled or unavailable providers.
- **Deterministic validation before model adjudication** on every quality-controlled path.
- **Judge → reviewer acceptance** with persisted evidence and rejection artifacts.
- **Exact and near-duplicate controls** appropriate to each dataset.
- **Accepted-token backfill for pretraining** after semantic review and final dedup.
- **Evaluation holdout protection** for SFT/DPO paths that use the holdout registry.
- **Consolidated Hugging Face publication** for each of the five dataset products.
- **Model qualification** that rejects mandatory-reasoning models and disables reasoning when supported.

## Getting Started

### Prerequisites

- Python 3 with `venv` support
- an OpenRouter API key for live generation
- a Hugging Face token only when publishing datasets

### Installation

~~~bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### Configuration

Copy the environment template:

~~~bash
cp .env.sample .env
~~~

Set at least:

~~~bash
OPENROUTER_API_KEY=...
HF_TOKEN=...              # only required for push commands
~~~

OpenRouter routing defaults to `auto`, which allows provider fallback. To prefer or pin a provider:

~~~bash
OPENROUTER_ROUTING_MODE=prefer OPENROUTER_PROVIDER=DeepInfra make sft-smoke
OPENROUTER_ROUTING_MODE=strict OPENROUTER_PROVIDER=DeepInfra make sft-smoke
~~~

Provider-order environment variables are also passed through by the Makefile:

~~~bash
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" make sft-smoke
~~~

### Usage

The Makefile is intentionally the supported entry point, but it is only a wrapper around the pipeline arguments. **Using a one-line Make target does not remove configurability.** Every important production setting can still be overridden with Make variables.

The three useful levels are:

1. default invocation;
2. a small set of targeted overrides;
3. a fully explicit production invocation.

The complete meaning of every supported variable is documented in [Parameter Reference](docs/PARAMETERS.md).

For users familiar with the previous direct Python CLI, the mapping is mechanical. For example, the Distillation-SFT command-line arguments became Make overrides:

| Previous pipeline argument | Current Make override |
|---|---|
| `--signals` / family selector | `DISTILLATION_SFT_SIGNALS` |
| `--seeds` | `DISTILLATION_SFT_SEEDS` |
| `--derivations-per-seed` | `DISTILLATION_SFT_DERIVATIONS_PER_SEED` |
| `--tasks-per-derivation` | `DISTILLATION_SFT_TASKS_PER_DERIVATION` |
| `--concurrency` | `DISTILLATION_SFT_CONCURRENCY` |
| `--answer-batch-size` | `DISTILLATION_SFT_ANSWER_BATCH_SIZE` |
| `--judge-batch-size` | `DISTILLATION_SFT_JUDGE_BATCH_SIZE` |
| `--reviewer-batch-size` | `DISTILLATION_SFT_REVIEWER_BATCH_SIZE` |
| `--derivation-model` | `DISTILLATION_SFT_DERIVATION_MODEL` |
| `--task-model` | `DISTILLATION_SFT_TASK_MODEL` |
| `--answer-model` | `DISTILLATION_SFT_ANSWER_MODEL` |
| `--judge-model` | `DISTILLATION_SFT_JUDGE_MODEL` |
| `--reviewer-model` | `DISTILLATION_SFT_REVIEWER_MODEL` |

The SFT/DPO/Distillation-DPO variables follow the same pattern with their dataset prefix.

#### Pretraining

Default smoke and production:

~~~bash
make pretrain-smoke
make pretrain-generate
~~~

Typical production override:

~~~bash
PRETRAIN_TARGET_RUN=pretrain-production-001 \
PRETRAIN_TARGET_TOKENS=1000000 \
PRETRAIN_TARGET_CONCURRENCY=4 \
make pretrain-generate
~~~

Fully explicit example:

~~~bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
PROFILE=balanced \
PRETRAIN_TARGET_RUN=pretrain-production-001 \
PRETRAIN_TARGET_TOKENS=1000000 \
PRETRAIN_BATCH_SIZE=32 \
PRETRAIN_TARGET_CONCURRENCY=4 \
PRETRAIN_MODEL=openai/gpt-5.6-luna-pro \
PRETRAIN_JUDGE_MODEL=google/gemma-4-31b-it \
PRETRAIN_REVIEWER_MODEL=openai/gpt-5.6-luna-pro \
PRETRAIN_JUDGE_BATCH_SIZE=10 \
PRETRAIN_REVIEWER_BATCH_SIZE=10 \
PRETRAIN_QUALITY_CONCURRENCY=8 \
PRETRAIN_STAGE_BATCH_ATTEMPTS=3 \
PRETRAIN_MAX_BACKFILL_ROUNDS=4 \
make pretrain-generate
~~~

Pretraining is different from the other four products: `PRETRAIN_TARGET_TOKENS` is the **final post-review, post-dedup accepted-token target**, not a raw generation budget.

#### Generic SFT

Default smoke and production:

~~~bash
make sft-smoke
make sft-generate
~~~

Typical focused run:

~~~bash
SFT_GENERATION_RUN=sft-programming-001 \
SFT_FAMILIES=programming \
SFT_CONCURRENCY=8 \
make sft-generate
~~~

Fully explicit production example:

~~~bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
SFT_GENERATION_RUN=sft-production-001 \
SFT_FAMILIES=all \
SFT_SEEDS=1 \
SFT_DERIVATIONS_PER_SEED=30 \
SFT_TASKS_PER_DERIVATION=15 \
SFT_CONCURRENCY=8 \
SFT_ANSWER_BATCH_SIZE=4 \
SFT_JUDGE_BATCH_SIZE=10 \
SFT_REVIEWER_BATCH_SIZE=10 \
SFT_CARDINALITY_FILL_ATTEMPTS=3 \
SFT_STAGE_BATCH_ATTEMPTS=3 \
SFT_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
SFT_TASK_MODEL=deepseek/deepseek-v4-flash \
SFT_ANSWER_MODEL=deepseek/deepseek-v4-flash \
SFT_JUDGE_MODEL=nvidia/nemotron-3.5-lightning \
SFT_REVIEWER_MODEL=google/gemma-4-31b-it \
SFT_DERIVATION_MAX_TOKENS=4096 \
SFT_TASK_MAX_TOKENS=4096 \
SFT_ANSWER_MAX_TOKENS=4096 \
SFT_JUDGE_MAX_TOKENS=4096 \
SFT_REVIEWER_MAX_TOKENS=512 \
SFT_JACCARD_THRESHOLD=0.82 \
SFT_SEQUENCE_THRESHOLD=0.90 \
make sft-generate
~~~

The default planned task volume per selected family is:

~~~text
SFT_SEEDS x SFT_DERIVATIONS_PER_SEED x SFT_TASKS_PER_DERIVATION
1 x 30 x 15 = 450 planned task candidates per family
~~~

Final accepted rows can be lower after novelty filtering, deterministic validation, judge/reviewer rejection, and final dedup.

#### Generic DPO

Default smoke and production:

~~~bash
make dpo-smoke
make dpo-generate
~~~

Typical focused run:

~~~bash
DPO_GENERATION_RUN=dpo-factual-001 \
DPO_PREFERENCE_DIMENSIONS=factual_accuracy \
DPO_CONCURRENCY=8 \
make dpo-generate
~~~

Fully explicit production example:

~~~bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
DPO_GENERATION_RUN=dpo-production-001 \
DPO_PREFERENCE_DIMENSIONS=all \
DPO_SEEDS=1 \
DPO_DERIVATIONS_PER_SEED=30 \
DPO_TASKS_PER_DERIVATION=15 \
DPO_CONCURRENCY=8 \
DPO_PAIR_BATCH_SIZE=4 \
DPO_JUDGE_BATCH_SIZE=10 \
DPO_REVIEWER_BATCH_SIZE=10 \
DPO_CARDINALITY_FILL_ATTEMPTS=3 \
DPO_STAGE_BATCH_ATTEMPTS=3 \
DPO_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
DPO_TASK_MODEL=deepseek/deepseek-v4-flash \
DPO_PAIR_MODEL=deepseek/deepseek-v4-flash \
DPO_JUDGE_MODEL=nvidia/nemotron-3.5-lightning \
DPO_REVIEWER_MODEL=google/gemma-4-31b-it \
DPO_DERIVATION_MAX_TOKENS=4096 \
DPO_TASK_MAX_TOKENS=4096 \
DPO_PAIR_MAX_TOKENS=4096 \
DPO_JUDGE_MAX_TOKENS=4096 \
DPO_REVIEWER_MAX_TOKENS=512 \
DPO_JACCARD_THRESHOLD=0.82 \
DPO_SEQUENCE_THRESHOLD=0.90 \
make dpo-generate
~~~

The default planned task volume is `1 x 30 x 15 = 450` preference tasks per selected dimension before quality losses.

#### Distillation SFT

Default smoke and production:

~~~bash
make distillation-sft-smoke
make distillation-sft-generate
~~~

Typical focused signal:

~~~bash
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-cloud-001 \
DISTILLATION_SFT_SIGNALS=cloud \
DISTILLATION_SFT_CONCURRENCY=8 \
make distillation-sft-generate
~~~

The explicit Make invocation is the direct replacement for the old long Python CLI. For example:

~~~bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
DISTILLATION_SFT_SIGNALS=cloud \
DISTILLATION_SFT_SEEDS=1 \
DISTILLATION_SFT_DERIVATIONS_PER_SEED=30 \
DISTILLATION_SFT_TASKS_PER_DERIVATION=15 \
DISTILLATION_SFT_CONCURRENCY=8 \
DISTILLATION_SFT_ANSWER_BATCH_SIZE=4 \
DISTILLATION_SFT_JUDGE_BATCH_SIZE=10 \
DISTILLATION_SFT_REVIEWER_BATCH_SIZE=10 \
DISTILLATION_SFT_CARDINALITY_FILL_ATTEMPTS=3 \
DISTILLATION_SFT_STAGE_BATCH_ATTEMPTS=3 \
DISTILLATION_SFT_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
DISTILLATION_SFT_TASK_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_SFT_ANSWER_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_SFT_JUDGE_MODEL=nvidia/nemotron-3.5-lightning \
DISTILLATION_SFT_REVIEWER_MODEL=google/gemma-4-31b-it \
DISTILLATION_SFT_DERIVATION_MAX_TOKENS=4096 \
DISTILLATION_SFT_TASK_MAX_TOKENS=4096 \
DISTILLATION_SFT_ANSWER_MAX_TOKENS=4096 \
DISTILLATION_SFT_JUDGE_MAX_TOKENS=4096 \
DISTILLATION_SFT_REVIEWER_MAX_TOKENS=512 \
DISTILLATION_SFT_JACCARD_THRESHOLD=0.82 \
DISTILLATION_SFT_SEQUENCE_THRESHOLD=0.90 \
make distillation-sft-generate
~~~

`DISTILLATION_SFT_TASK_MODEL` writes the **student-facing prompt**. `DISTILLATION_SFT_ANSWER_MODEL` is the teacher that writes the response used for downstream response distillation.

#### Distillation DPO

Default smoke and production:

~~~bash
make distillation-dpo-smoke
make distillation-dpo-generate
~~~

Typical focused dimension:

~~~bash
DISTILLATION_DPO_TARGET_RUN=distillation-dpo-code-001 \
DISTILLATION_DPO_DIMENSIONS=code_correctness \
DISTILLATION_DPO_CONCURRENCY=8 \
make distillation-dpo-generate
~~~

Fully explicit production example:

~~~bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
DISTILLATION_DPO_TARGET_RUN=distillation-dpo-production-001 \
DISTILLATION_DPO_DIMENSIONS=all \
DISTILLATION_DPO_SEEDS=1 \
DISTILLATION_DPO_DERIVATIONS_PER_SEED=30 \
DISTILLATION_DPO_TASKS_PER_DERIVATION=15 \
DISTILLATION_DPO_CONCURRENCY=8 \
DISTILLATION_DPO_PAIR_BATCH_SIZE=4 \
DISTILLATION_DPO_JUDGE_BATCH_SIZE=10 \
DISTILLATION_DPO_REVIEWER_BATCH_SIZE=10 \
DISTILLATION_DPO_CARDINALITY_FILL_ATTEMPTS=3 \
DISTILLATION_DPO_STAGE_BATCH_ATTEMPTS=3 \
DISTILLATION_DPO_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
DISTILLATION_DPO_TASK_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_DPO_PAIR_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_DPO_JUDGE_MODEL=google/gemma-4-31b-it \
DISTILLATION_DPO_REVIEWER_MODEL=openai/gpt-5.6-luna-pro \
DISTILLATION_DPO_DERIVATION_MAX_TOKENS=4096 \
DISTILLATION_DPO_TASK_MAX_TOKENS=4096 \
DISTILLATION_DPO_PAIR_MAX_TOKENS=4096 \
DISTILLATION_DPO_JUDGE_MAX_TOKENS=4096 \
DISTILLATION_DPO_REVIEWER_MAX_TOKENS=512 \
DISTILLATION_DPO_JACCARD_THRESHOLD=0.82 \
DISTILLATION_DPO_SEQUENCE_THRESHOLD=0.90 \
make distillation-dpo-generate
~~~

Distillation DPO intentionally uses Gemma as judge and Luna as reviewer because its quality contract is the stricter five-gate distillation preference check.

### Inspect, Report, and Publish

Generation targets automatically build their report. You can rerun inspection/reporting explicitly:

~~~bash
SFT_INSPECT_RUN=sft-production-001 make sft-inspect
SFT_REPORT_RUN=sft-production-001 make sft-report
~~~

Publication always names the run explicitly when it differs from the reporting default:

~~~bash
SFT_PUSH_RUN=sft-production-001 \
SFT_HF_REPO=tohio/slm-synthetic-sft \
make sft-push
~~~

Before changing to an unqualified model, run the appropriate qualification target:

~~~bash
QUALIFY_MODEL=deepseek/deepseek-v4-flash make model-qualify-sft
~~~

For the exhaustive variable reference—including run ids, output roots, model roles, token ceilings, batching, concurrency, retry behavior, novelty thresholds, holdouts, and publication controls—see [Parameter Reference](docs/PARAMETERS.md).

## Project Structure

~~~text
.
├── configs/                    # pretraining configuration template and holdout registry
├── docs/                       # architecture, workflow, commands, dataset reference
├── slm_synth/
│   ├── runtime/                # shared execution mechanics
│   ├── pretrain/               # synthetic pretraining dataset
│   ├── sft/                    # generic supervised fine-tuning dataset
│   ├── dpo/                    # generic preference dataset
│   ├── distillation_sft/       # teacher-response distillation dataset
│   ├── distillation_dpo/       # post-distillation preference dataset
│   └── taxonomy/               # shared public metadata and holdout definitions
├── tests/                      # unit/integration contract tests
├── Makefile                    # supported user-facing command surface
└── requirements.txt
~~~

Each non-trivial package contains its own README describing its internal ownership and conventions.

## Documentation

Start with the [documentation index](docs/README.md).

For normal operation, the most useful documents are:

- [Command Reference](docs/COMMANDS.md)
- [Parameter Reference](docs/PARAMETERS.md)
- [Generation Workflow](docs/GENERATION_WORKFLOW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset Purpose and Contracts](docs/DATASET_PURPOSE.md)
- [Generation Families and Dimensions](docs/GENERATION_FAMILIES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Testing

Run the repository test suite with:

~~~bash
make test
~~~

For focused development:

~~~bash
pytest -q tests/test_sft_*.py
pytest -q tests/test_dpo_*.py
pytest -q tests/test_distillation_*.py
pytest -q tests/test_pretrain_*.py
~~~

Tests do not replace live smoke runs. Provider routing, structured-output support, model suitability, and real generation quality must still be validated with the appropriate smoke target.

## Status

The supported architecture is:

**five datasets → one shared runtime → one production path per dataset**.

Legacy generation/orchestration command surfaces are not supported.

## License

MIT. See [LICENSE](LICENSE).
