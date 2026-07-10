# `slm_synth/distillation_dpo`

## Purpose

This package owns distillation-specific preference dataset generation. It builds deterministic source specs, requests structured teacher preference rows, applies pair-quality gates, writes public JSONL files, records lineage manifests, reports coverage, and publishes `distillation-dpo-*` artifacts.

It does not produce generic DPO data, response-distillation rows, student-sampled pairs, or model-training artifacts.

## Contents

```text
distillation_dpo/
├── seeds.py          # family definitions
├── spec_builders.py  # source specs for teacher generation
├── batches.py        # batch prompt and response contract
├── pair_quality.py   # pair-quality gates
├── schema.py         # public row validation
├── io.py             # JSONL and manifest writers
├── runs.py           # multi-family LLM run orchestration
├── report.py         # coverage reporting
├── card.py           # dataset card rendering
├── push_hf.py        # Hugging Face publishing
└── cli.py            # command-line entrypoint
```

## How It Fits In

`distillation-dpo-*` artifacts are consumed by `slm-distillation` after response distillation when DPO alignment is needed. Generic DPO remains under `../dpo/`.

## Conventions

Manifests record:

```yaml
dataset_type: distillation-dpo
chosen_source: teacher
rejected_source: controlled_weak
target_consumer: slm-distillation
```

Public rows omit provider, retry, cost, and run internals.
