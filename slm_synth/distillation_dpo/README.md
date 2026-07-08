# `slm_synth/distillation_dpo`

## Purpose

This package owns distillation-specific preference pairs for aligning distilled models. It materializes teacher-quality chosen responses and controlled-weak rejected responses, applies pair-quality gates, writes public JSONL files, records lineage manifests, reports coverage, and publishes `distillation-dpo-*` artifacts.

It does not call the teacher live for student sampling, replace generic DPO, or train distilled models.

## Contents

```text
distillation_dpo/
├── seeds.py          # smoke families and seed pairs
├── spec_builders.py  # deterministic production pair builders
├── pair_quality.py   # pair-quality gates
├── schema.py         # public row validation
├── io.py             # JSONL and manifest writers
├── runs.py           # smoke and production run orchestration
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
