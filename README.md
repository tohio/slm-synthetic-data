# SLM Synthetic Data

Synthetic dataset generation for the SLM training stack.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository generates synthetic datasets for pretraining, supervised fine-tuning, preference tuning, and distillation workflows in the SLM stack. It owns data creation, validation, manifests, coverage reports, dataset cards, and Hugging Face publishing helpers. It does not train models, run trained-model evals, create checkpoints, export models, or produce logits artifacts.

## Architecture

![Architecture](docs/architecture.png)

OpenRouter-backed generation flows through `slm_synth/llm.py` for provider calls, retries, routing, structured outputs, and telemetry. Dataset-specific packages own their public row contracts, reports, and push surfaces. Pretraining uses grounded artifact builders; SFT, DPO, and distillation SFT use structured provider generation; distillation DPO uses deterministic teacher-quality versus controlled-weak preference builders.

## Features

- Grounded synthetic pretraining signals.
- Generic SFT and DPO dataset generation.
- Distillation SFT teacher prompt/response datasets.
- Distillation DPO preference datasets for distilled-model alignment.
- Adaptive request admission, adaptive batch sizing, retry telemetry, and OpenRouter routing policy.
- Public dataset rows separated from manifests, scratch files, provider internals, batch shards, and retry artifacts.

## Getting Started

### Prerequisites

- Python 3.12 or newer.
- An OpenRouter API key for live generation.
- A Hugging Face token only when publishing datasets.

### Installation

```bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

The default live model is configured by `MODEL` and can be overridden on generation commands:

```bash
MODEL=openai/gpt-4.1-mini
```

### Smoke Run

Run a cheap SFT smoke job before any paid target run:

```bash
make sft-smoke
make sft-inspect SFT_INSPECT_RUN=sft-smoke-001
```

### Small Generation Run

Use a small target override before production:

```bash
SFT_TARGET_ROWS=100 SFT_TARGET_RUN=sft-small-001 make sft-generate
```

Inspect the generated public rows and run manifest:

```bash
make sft-inspect SFT_INSPECT_RUN=sft-small-001
make sft-report SFT_REPORT_RUN=sft-small-001
```

Generated files are written under:

```text
data/sft/runs/sft-small-001/
  datasets/    public JSONL files
  manifests/   run and batch manifests
  batches/     internal batch shards
```

### Production Run

After smoke and small-scale outputs pass inspection, run the full SFT target:

```bash
SFT_TARGET_ROWS=14000 SFT_TARGET_RUN=sft-prod-001 make sft-generate
```

Push only after inspecting the public dataset files and manifests:

```bash
make sft-push SFT_PUSH_RUN=sft-prod-001
```

For end-to-end workflows across every generation surface, see `docs/GENERATION_WORKFLOW.md`. For Make target details, see `docs/COMMANDS.md`.

## Project Structure

```text
.
├── configs/             # generation config templates and helpers
├── docs/                # workflow, command, dataset, disk, and architecture docs
├── slm_synth/           # Python package for generation and publishing
├── tests/               # unit and integration tests
├── Makefile             # supported command surface
└── README.md            # project overview
```

Important package folders have their own READMEs under `slm_synth/`.

## Documentation

See `docs/README.md` for the documentation index.

## Testing

```bash
make test
```

Run focused tests while changing one artifact surface:

```bash
pytest -q tests/test_sft_*.py
pytest -q tests/test_dpo_*.py
pytest -q tests/test_distillation_*.py
```

## Status

The repository is ready for the run ladder: smoke runs, validation checks, small-scale target overrides, then full production runs after outputs pass inspection.

## License

MIT. See [LICENSE](LICENSE).
