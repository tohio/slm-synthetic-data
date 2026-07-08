# SLM Synthetic Data

Synthetic dataset generation for the SLM training stack.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository generates synthetic datasets for pretraining, supervised fine-tuning, preference tuning, and distillation workflows in the SLM stack. It owns data creation, validation, manifests, coverage reports, dataset cards, and Hugging Face publishing helpers. It does not train models, run trained-model evals, create checkpoints, export models, or produce logits artifacts.

## Architecture

![Architecture](docs/architecture.png)

OpenRouter-backed generation flows through `slm_synth/llm.py` for provider calls, retries, routing, structured outputs, and telemetry. Dataset-specific packages own their public row contracts, reports, and push surfaces. Pretraining uses grounded artifact/source builders; SFT, DPO, and distillation SFT use structured provider generation; distillation DPO uses deterministic teacher-quality versus controlled-weak preference builders.

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

The default live model is configured by `MODEL`:

```bash
MODEL=openai/gpt-4.1-mini
```

### Usage

Print the command surface:

```bash
make help
```

Run smoke jobs before any larger target run:

```bash
make pretrain-smoke
make sft-smoke
make dpo-smoke
make distillation-sft-smoke
make distillation-dpo-smoke
```

Inspect generated public artifacts:

```bash
make pretrain-inspect
make sft-inspect
make dpo-inspect
make distillation-sft-inspect
make distillation-dpo-inspect
```

## Project Structure

```text
.
├── configs/             # generation config templates and helpers
├── docs/                # command, dataset, disk, and architecture docs
├── prompts/             # pretraining prompt templates
├── slm_synth/           # Python package for generation and publishing
├── tests/               # unit and integration tests
├── Makefile             # supported command surface
└── README.md            # project overview
```

Important package folders have their own READMEs under `slm_synth/`.

## Documentation

See `docs/README.md` for command reference, dataset contracts, and setup notes.

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

MIT. See `LICENSE`.
