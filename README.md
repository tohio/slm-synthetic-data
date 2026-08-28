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

Start with a smoke run for the dataset you intend to generate:

~~~bash
make pretrain-smoke
make sft-smoke
make dpo-smoke
make distillation-sft-smoke
make distillation-dpo-smoke
~~~

Inspect the generated rows and manifests:

~~~bash
make sft-inspect
make sft-report
~~~

Before changing to an unqualified model, run the corresponding qualification target:

~~~bash
QUALIFY_MODEL=deepseek/deepseek-v4-flash make model-qualify-sft
~~~

Then run the production command:

~~~bash
SFT_GENERATION_RUN=sft-production-001 make sft-generate
~~~

Publication is explicit:

~~~bash
SFT_PUSH_RUN=sft-production-001 make sft-push
~~~

See [Command Reference](docs/COMMANDS.md) for every supported variable and example.

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
