# `slm_synth/dpo`

## Purpose

This package generates generic teacher-backed DPO preference datasets. It owns source specifications, pair generation, output acceptance, replacement rounds, safe resume, coverage reporting, manifests, dataset cards, and Hugging Face publication.

It does not produce SFT rows, distillation-specific pairs, pretraining records, or model-training artifacts.

## Contents

```text
dpo/
├── spec_builders.py  # capacity-bounded source specifications
├── specs.py          # teacher-visible specification validation
├── batches.py        # batch prompts and response validation
├── generation.py     # teacher-backed batch generation
├── acceptance.py     # normalized uniqueness and pair-quality reporting
├── runs.py           # multi-family generation, backfill, and resume
├── schema.py         # public row validation
├── manifest.py       # dataset and run manifests
├── report.py         # acceptance, holdout, and coverage reporting
├── card.py           # consolidated dataset-card configuration validation
├── push_hf.py        # one-repository Hugging Face publication
└── cli.py            # command-line entrypoint
```

## Generation

DPO is organized by ten preference dimensions. Each pair also carries the shared task, interaction, output, and context axes used by generic SFT. The chosen and rejected responses are generated semantically by the configured teacher; no eval-shaped family or deterministic answer-pair path remains.

`source_catalog.py` owns nine independently authored prompts per preference
dimension. It does not depend on SFT specs. The full 90-prompt catalog is
checked for semantic-source uniqueness, near-duplicates, template
concentration, and SFT prompt overlap before generation can construct a
provider backend. Requested source ranges, including the configured replacement
budget, are also validated. Model selection remains configurable through
`DPO_MODEL`; OpenRouter routing remains configurable through the shared routing
variables.

## Acceptance and Resume

Public output keeps the first pair for each normalized ID, prompt, and `(prompt, chosen, rejected)` triple. Duplicate or locally rejected pairs do not count toward the accepted target. Replacement rounds use new source indexes and preserve accepted pairs.

An exhausted run writes its accepted files and an underfilled manifest, then exits nonzero. Resume verifies the source plan, family allocation, accepted-file fingerprints, batch manifests, accounting, and next unused source indexes before generating replacements.

Coverage reports include aggregate and per-family metadata, uniqueness, chosen/rejected similarity, negative-pattern distribution, holdout results, and attempted/accepted/rejected/duplicate/remaining counts. Similarity and repeated negative patterns are diagnostics; exact duplicates, holdout collisions, stale accounting, and underfilled targets are publication blockers.

## Publication

`dpo-push` publishes one complete run to `DPO_HF_REPO` in one commit:

```text
README.md
data/<family>.jsonl
artifacts/coverage.json
artifacts/manifests/*.manifest.json
```

The default dataset configuration loads all family files. Named configurations load one family without duplicating stored pairs. Publication requires one final JSONL file per manifest family and a current publish-ready report and dataset card.

## Commands

Use `make dpo-smoke`, `make dpo-generate`, `make dpo-report`, `make dpo-inspect`, and `make dpo-push`. See `../../docs/COMMANDS.md` for variables and `../../docs/GENERATION_WORKFLOW.md` for the run ladder.
