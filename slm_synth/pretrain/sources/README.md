# `slm_synth/pretrain/sources`

## Purpose

This folder renders pretraining artifacts into grounded generation records. Each source module maps deterministic artifact objects to prompts, expected anchors, and validation metadata for one signal family.

It does not define the source facts themselves; those live under `../artifacts/`.

## Contents

```text
sources/
├── arithmetic.py
├── task_code.py
├── educational_qa_mcq_math.py
├── educational_qa_mcq_general.py
├── factual_restraint.py
└── two_pass.py
```

## How It Fits In

The pretraining generator calls these source modules to build promptable records. Outputs are validated by `pretrain/validate.py` and summarized by pretraining manifest/report helpers.

## Conventions

Source renderers should keep enough anchor metadata for local validation and audits. Avoid embedding provider-specific retry or transport logic in source modules.
