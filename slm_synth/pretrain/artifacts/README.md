# `slm_synth/pretrain/artifacts`

## Purpose

This folder defines deterministic source artifacts for grounded pretraining generation. Artifacts are local facts, tasks, examples, and constraints used to render provider prompts with known expected structure.

It does not call providers or write datasets directly.

## Contents

```text
artifacts/
├── arithmetic.py                    # verified integer arithmetic artifacts
├── task_code.py                     # Python task/code artifacts
├── educational_qa_mcq_math.py       # math multiple-choice artifacts
├── educational_qa_mcq_general.py    # general educational MCQ artifacts
├── factual_restraint.py             # restraint and uncertainty artifacts
├── lexicon.py                       # shared vocab/source lists
├── quality.py                       # artifact quality checks
└── base.py                          # shared artifact types/helpers
```

## How It Fits In

`slm_synth/pretrain/sources/` renders these artifacts into generation records. Preflight and report commands inspect this folder to catch duplicate or low-quality source material before provider calls.

## Conventions

Artifacts should be deterministic, inspectable, and cheap to validate locally. Add provider-facing rendering logic under `pretrain/sources/`, not here.
