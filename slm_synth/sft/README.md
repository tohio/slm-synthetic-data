# `slm_synth/sft`

## Purpose

This package owns generic supervised fine-tuning dataset generation. It builds task specs, requests teacher-authored conversation content through a portable plain-text contract, validates public SFT rows, writes family JSONL files, records manifests, reports coverage, and publishes public artifacts.

It does not produce DPO preference pairs, response-distillation rows, deterministic seed datasets, or model-training artifacts.

## Contents

```text
sft/
├── spec_builders.py  # scalable family spec builders
├── specs.py          # teacher-visible spec validation
├── batches.py        # batch prompt and response contract
├── adjudication.py   # independent judge and reviewer gates
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

Make targets `sft-smoke`, `sft-generate`, `sft-report`, `sft-inspect`, and
`sft-push` call this package. `sft-push` publishes one repository containing
one flat `data/<task_family>.jsonl` file per selected family, a default
all-family configuration, and one named configuration per family. Batch shards
and nested JSONL files are never export candidates, and `generate-llm-run` is
the only generic SFT generation command. Public command details live in
`../../docs/COMMANDS.md`.

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

Live candidates are rendered from the complete grounded brief. Models return
only language-bearing fields; repository code attaches IDs, metadata, taxonomy,
tools, and run fields. Calls use ordinary messages and max output only by
default—no required response schema, tool choice, reasoning, temperature, or
top-p parameter. Local code then checks schema, interaction, tool, and
output-mode compliance.
Finite briefs may also declare internal machine-checkable word, line, item,
term, heading, and JSON-key constraints. These run before semantic
adjudication and are recorded per accepted row in batch manifests.
Critical taxonomies and source facts may additionally declare exact public
prompt requirements, preventing a rendered answer from relying on hidden brief
material. Source-specific quality requirements are adjudicated independently.
After deterministic validation, a conservative judge either accepts or rejects
each candidate and must reject ambiguity, insufficient evidence, or anything it
cannot assess reliably. Only judge-accepted candidates reach a reviewer. The
reviewer answers whether the judge's acceptance was justified. Final acceptance
requires deterministic success, judge acceptance, and reviewer agreement.
Semantic rejection is final and never causes quota backfill. Malformed or
transient provider responses retain bounded retry, backoff, and adaptive
controls. `SFT_ADJUDICATOR_MODEL` and `SFT_REVIEWER_MODEL` are independently
configurable.

Publication re-audits the accepted public rows rather than trusting generation
alone. It blocks exact or near-duplicate prompts and conversations, repeated
assistant-response clusters, templates exceeding 40% of accepted rows,
malformed role or tool sequences, and holdout collisions. When a run manifest
is supplied, every public row ID must also map to a passing semantic decision
and deterministic output decision in a referenced batch manifest; missing,
malformed, or failed evidence blocks publication. The Hugging Face push repeats all file-derived checks against the
live JSONL files so a stale clean report cannot bypass them.
