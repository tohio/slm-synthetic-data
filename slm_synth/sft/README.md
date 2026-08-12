# `slm_synth/sft`

## Purpose

This package owns generic supervised fine-tuning dataset generation. It builds task specs, requests structured teacher chat rows, validates public SFT rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce DPO preference pairs, response-distillation rows, deterministic seed datasets, or model-training artifacts.

## Contents

```text
sft/
├── spec_builders.py  # scalable family spec builders
├── specs.py          # teacher-visible spec validation
├── batches.py        # batch prompt and response contract
├── generation.py     # one-batch materialization/generation
├── acceptance.py     # normalized output uniqueness and acceptance
├── runs.py           # multi-family generation, backfill, and resume
├── schema.py         # public row validation
├── manifest.py       # dataset and run manifests
├── report.py         # aggregate and per-family acceptance reporting
├── card.py           # consolidated dataset configuration validation
├── push_hf.py        # atomic consolidated Hugging Face publishing
└── cli.py            # command-line entrypoint
```

## How It Fits In

Make targets `sft-smoke`, `sft-generate`, `sft-report`, `sft-inspect`, and `sft-push` call this package. `sft-push` publishes one repository containing every family file, a default all-family configuration, and optional per-family configurations. Public command details live in `../../docs/COMMANDS.md`.

## Conventions

Public SFT rows contain only `id`, `messages`, and public `metadata`. Teacher/provider/run/cost/retry details stay in manifests.

Accepted targets count unique public rows after local validation. Backfill uses unused source indexes; finalized underfilled runs can be continued with `SFT_RESUME=true`. The default Make paths enforce the configured holdout registry during generation and reporting.
