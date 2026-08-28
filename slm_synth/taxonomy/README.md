# `slm_synth/taxonomy`

## Purpose

Define shared public metadata labels and evaluation holdout behavior used across alignment/distillation datasets.

This package does not generate model content.

## Contents

~~~text
taxonomy/
├── alignment_metadata.py
├── categories.py
├── context_modes.py
├── difficulties.py
├── eval_families.py
├── failure_modes.py
├── holdouts.py
├── interaction_modes.py
├── metadata.py
├── output_modes.py
├── preference_dimensions.py
├── task_families.py
└── template_families.py
~~~

## Key Files

| File | Purpose |
|---|---|
| `metadata.py` | Validates public taxonomy metadata combinations. |
| `holdouts.py` | Loads and checks evaluation prompt/key exclusions. |
| `eval_families.py` | Shared evaluation-family vocabulary. |
| `template_families.py` | Shared structural/template labels. |
| `failure_modes.py` | Preference/error labels, especially for DPO products. |

## How It Fits In

SFT/DPO/distillation packages use taxonomy labels in public `metadata` and reports. Centralizing common labels prevents equivalent concepts from drifting across products.

See [Dataset Purpose and Contracts](../../docs/DATASET_PURPOSE.md).

## Conventions

- Add a taxonomy label here only when multiple dataset products genuinely share the concept.
- Keep dataset-private generation variables inside the owning package.
- Same-family generation is allowed when it does not reproduce an exact evaluation prompt or registered holdout key.
- Do not use the holdout registry as a broad topical deny-list.
