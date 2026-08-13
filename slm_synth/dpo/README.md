# `slm_synth/dpo`

## Purpose

This package owns generic DPO preference dataset generation. It builds task specs, requests structured teacher preference rows, validates public DPO rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce SFT chat rows, deterministic seed datasets, distillation-specific DPO pairs, or model-training artifacts.

## Contents

```text
dpo/
├── spec_builders.py  # capacity-bounded specs and controlled negatives
├── specs.py          # teacher-visible spec validation
├── batches.py        # batch prompt and response contract
├── generation.py     # per-spec deterministic or teacher generation
├── runs.py           # multi-family LLM run orchestration
├── schema.py         # public row validation
├── manifest.py       # dataset and run manifests
├── report.py         # coverage reporting
├── push_hf.py        # Hugging Face publishing
└── cli.py            # command-line entrypoint
```

## How It Fits In

Make targets `dpo-smoke`, `dpo-generate`, `dpo-report`, `dpo-inspect`, and `dpo-push` call this package. Public command details live in `../../docs/COMMANDS.md`.

## Conventions

Public DPO rows contain `id`, `prompt`, `chosen`, `rejected`, and public `metadata`. Generic DPO stays separate from `distillation_dpo`, which has different lineage and consumer metadata.

Every family inherits the unique source capacity of its SFT source family. Arithmetic, factual answer-only, expression, exact-format, and verifiable code families are materialized locally from exact targets. Concept explanations, code explanations, and private-fact restraint remain teacher-generated. Mixed batches send only teacher-required specifications to the provider.

At the approved 1,000-pair family target, source specifications are unique for all 14 families and normalized deterministic triples are unique for all 11 exact-target families. A requested source range that exceeds declared capacity fails before provider-backend construction.
