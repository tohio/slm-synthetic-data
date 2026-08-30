# `slm_synth/pretrain/artifacts`

## Purpose

Define deterministic, inspectable source artifacts used to ground synthetic pretraining generation.

Artifact factories decide the underlying facts/tasks/constraints. They do not call providers and do not write the final training dataset.

## Contents

~~~text
artifacts/
├── base.py
├── arithmetic.py
├── task_code.py
├── task_code_catalog.py
├── educational_qa_mcq_math.py
├── educational_qa_mcq_general.py
├── factual_restraint.py
├── lexicon.py
└── quality.py
~~~

## Key Files

| File | Purpose |
|---|---|
| `base.py` | Shared grounded artifact representation. |
| `arithmetic.py` | Locally verifiable integer/numerical backbones. |
| `task_code.py` / `task_code_catalog.py` | Deterministic programming tasks and catalog. |
| `educational_qa_mcq_math.py` | Math questions with locally known answer structure. |
| `educational_qa_mcq_general.py` | Self-contained educational questions grounded in supplied evidence/rules. |
| `factual_restraint.py` | Cases where uncertainty/non-invention is known by construction. |
| `quality.py` | Deterministic artifact-level quality checks. |

## How It Fits In

`pretrain/grounded.py` renders these artifacts into provider-facing prompts. The generated record is then deterministically checked against what the artifact makes verifiable before semantic judge/reviewer stages.

See [Pretraining Package](../README.md).

## Conventions

- Artifact generation should be deterministic for a given index/config.
- Artifacts should contain enough local information to validate the generated record.
- Provider-facing prose belongs in `pretrain/grounded.py`, not in the artifact factory.
- Do not add examples copied from evaluation prompts.
- The finite catalogs are seed archetypes and preflight coverage, not the full production inventory.
- Production indexes beyond the seed inventory use deterministic derivation profiles; derived rows must be materially distinct rather than renamed, renumbered, entity-swapped, or cosmetically paraphrased seeds.
- Deterministic validation plus judge/reviewer acceptance remains mandatory for every derived row.
- Favor large combinatorial capacity over repeated paraphrase templates.
