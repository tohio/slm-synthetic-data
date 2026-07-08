# `slm_synth/taxonomy`

## Purpose

This package defines shared public metadata labels used by SFT, DPO, distillation DPO, and related reports. It keeps category, difficulty, eval-family, template-family, failure-mode, and holdout labels consistent across dataset families.

It does not generate rows or call providers.

## Contents

```text
taxonomy/
├── categories.py
├── difficulties.py
├── eval_families.py
├── failure_modes.py
├── holdouts.py
├── metadata.py
└── template_families.py
```

## How It Fits In

Dataset packages use these labels in public `metadata` fields and coverage reports. Keeping the taxonomy centralized prevents SFT and DPO families from drifting.

## Conventions

Add labels here when they are shared across dataset families. Keep family-specific private variables inside the owning dataset package.
