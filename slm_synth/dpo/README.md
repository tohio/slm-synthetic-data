# `slm_synth/dpo`

## Purpose

This package owns generic DPO preference dataset generation. It builds task specs, requests structured teacher preference rows, validates public DPO rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce SFT chat rows, distillation-specific DPO pairs, or model-training artifacts.

## Contents

```text
dpo/
├── spec_builders.py  # scalable family spec builders
├── specs.py          # teacher-visible spec validation
├── batches.py        # batch prompt and response contract
├── generation.py     # one-batch materialization/generation
├── runs.py           # multi-family run orchestration
├── schema.py         # public row validation
├── seeds.py          # deterministic smoke rows
├── manifest.py       # dataset and run manifests
├── report.py         # coverage reporting
├── push_hf.py        # Hugging Face publishing
└── cli.py            # command-line entrypoint
```

## How It Fits In

Make targets `dpo-smoke`, `dpo-generate`, `dpo-report`, `dpo-inspect`, and `dpo-push` call this package. Public command details live in `../../docs/COMMANDS.md`.

## Conventions

Public DPO rows contain `id`, `prompt`, `chosen`, `rejected`, and public `metadata`. Generic DPO stays separate from `distillation_dpo`, which has different lineage and consumer metadata.
