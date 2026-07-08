# `slm_synth`

## Purpose

`slm_synth` owns synthetic dataset generation for the SLM stack. It contains provider integration, adaptive request control, validators, run manifests, coverage reports, and Hugging Face publishing helpers.

It does not train models, run trained-model evals, create checkpoints, export models, or build logits artifacts.

## Contents

```text
slm_synth/
├── pretrain/            # grounded synthetic pretraining records
├── sft/                 # generic supervised fine-tuning datasets
├── dpo/                 # generic preference datasets
├── distillation_sft/    # teacher prompt/response datasets for distillation
├── distillation_dpo/    # preference pairs for distilled-model alignment
├── taxonomy/            # shared public metadata labels
├── adaptive_batch.py    # adaptive batch-size controller
├── llm.py               # OpenRouter client, retries, telemetry, routing
├── planning.py          # target-count allocation helpers
└── push_hf.py           # shared Hugging Face utilities
```

## Key Files

| File | Purpose |
|---|---|
| `llm.py` | OpenRouter structured generation, retry handling, routing policy, and provider telemetry. |
| `adaptive_batch.py` | Batch-size backoff/ramp behavior after provider or parse failures. |
| `planning.py` | Even target-count allocation across selected families/signals. |
| `telemetry.py` | Run-level aggregation of per-batch/provider telemetry. |
| `paths.py` | Shared path helpers. |
| `push_hf.py` | Shared upload helpers used by dataset-specific push modules. |

## How It Fits In

Dataset-specific packages expose Make/CLI surfaces for pretraining, SFT, DPO, distillation SFT, and distillation DPO artifacts. Public command documentation lives in `../docs/COMMANDS.md`; dataset contracts live in `../docs/DATASET_PURPOSE.md`.

## Conventions

- Public row contracts stay inside the dataset package that owns them.
- Provider, cost, retry, routing, and teacher lineage details belong in manifests, not public rows.
- Public artifact names use hyphens, such as `distillation-sft-*`; Python package names use underscores, such as `slm_synth.distillation_sft`.
