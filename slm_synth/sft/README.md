# `slm_synth/sft`

## Purpose

This package owns generic supervised fine-tuning dataset generation. It builds task specs, requests structured teacher chat rows, validates public SFT rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce DPO preference pairs, response-distillation rows, deterministic seed datasets, or model-training artifacts.

## Contents

```text
sft/
├── spec_builders.py  # scalable family spec builders
├── specs.py          # teacher-visible spec validation
├── batches.py        # batch prompt and response contract
├── adjudication.py   # independent semantic quality gate
├── generation.py     # one-batch materialization/generation
├── acceptance.py     # normalized output uniqueness and acceptance
├── runs.py           # multi-family candidate generation and acceptance
├── schema.py         # public row validation
├── manifest.py       # dataset and run manifests
├── report.py         # aggregate and per-family acceptance reporting
├── card.py           # consolidated dataset configuration validation
├── push_hf.py        # atomic consolidated Hugging Face publishing
└── cli.py            # command-line entrypoint
```

## How It Fits In

Make targets `sft-smoke`, `sft-generate`, `sft-report`, `sft-inspect`, and `sft-push` call this package. `sft-push` publishes one repository containing every family file, a default all-family configuration, and optional per-family configurations. Public command details live in `../../docs/COMMANDS.md`.

## Conventions

Public SFT rows contain `id`, strict multi-turn `messages`, public `metadata`,
and optional shared `tools`. A system message is permitted only at the start.
Assistant tool calls use structured argument objects; tool responses must
resolve every call ID before a final assistant response. Adjacent roles,
undeclared tools, duplicate calls, and unresolved calls fail validation rather
than being normalized. Teacher/provider/run/cost/retry details stay in
manifests.

Production runs require explicit candidate counts for every selected family. Candidate counts limit generation work; accepted rows are the quality-filtered outcome and are not backfilled to reach a quota. The default Make paths enforce the configured holdout registry during generation and reporting.
Manifests and coverage reports include accepted-row counts and a tokenizer-independent estimate of the public chat payload at four characters per token. Downstream tokenization remains authoritative.

`source_catalog.py` declares six genuinely different briefs for each of the ten
task families. `python -m slm_synth.alignment_preflight --kind sft` validates
the complete 60-source catalog, semantic-source uniqueness, near-duplicates,
template concentration, and axis coverage. Run orchestration calls this gate
before constructing a paid teacher backend.

Live candidates are rendered from the complete grounded brief, then checked
locally for schema, metadata, interaction, tool, and output-mode compliance.
An independent structured adjudication call scores correctness, grounding,
instruction adherence, completeness, and coherence and verifies every source
constraint. A candidate is written only when every score is at least 3/4 and
every constraint passes. `SFT_ADJUDICATOR_MODEL` and
`SFT_ADJUDICATOR_MAX_TOKENS` default to the renderer settings but can be
overridden; both roles retain the same routing, retry, backoff, and adaptive
request controls.
