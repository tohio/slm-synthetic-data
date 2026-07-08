# `slm_synth/sft`

## Purpose

This package owns generic supervised fine-tuning dataset generation. It builds task specs, requests structured teacher chat rows, validates public SFT rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce DPO preference pairs, response-distillation rows, or model-training artifacts.

## Contents

```text
sft/
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

Make targets `sft-smoke`, `sft-generate`, `sft-report`, `sft-inspect`, and `sft-push` call this package. Public command details live in `../../docs/COMMANDS.md`.

## Conventions

Public SFT rows contain only `id`, `messages`, and public `metadata`. Teacher/provider/run/cost/retry details stay in manifests.
