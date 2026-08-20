# `slm_synth/dpo`

## Purpose

This package generates generic teacher-backed DPO preference datasets. It owns finite source specifications, pair generation, quality acceptance, coverage reporting, manifests, dataset cards, and Hugging Face publication.

It does not produce SFT rows, distillation-specific pairs, pretraining records, or model-training artifacts.

## Contents

```text
dpo/
├── spec_builders.py  # capacity-bounded source specifications
├── specs.py          # teacher-visible specification validation
├── batches.py        # batch prompts and response validation
├── adjudication.py   # preference-quality and separation gate
├── generation.py     # teacher-backed batch generation
├── acceptance.py     # normalized uniqueness and pair-quality reporting
├── runs.py           # candidate-budgeted multi-dimension generation
├── schema.py         # public row validation
├── manifest.py       # dataset and run manifests
├── report.py         # acceptance, holdout, and coverage reporting
├── card.py           # consolidated dataset-card configuration validation
├── push_hf.py        # one-repository Hugging Face publication
└── cli.py            # command-line entrypoint
```

## Generation

DPO is organized by ten preference dimensions. Each pair also carries the shared task, interaction, output, and context axes used by generic SFT.

Live generation is staged. The renderer first creates the shared prompt and a
high-quality chosen branch. A second call receives that candidate and
introduces exactly one plausible weakness matching `preference_dimension` and
`failure_mode`. An independent adjudicator then checks chosen quality, rejected
plausibility, weakness match, preference separation, collateral preservation,
and every source constraint. No local correct-number/wrong-number fabrication
or copied-branch repair remains.

Public rows preserve one explicit `prompt`, `chosen`, and `rejected`. Optional
tool definitions occur once at row level and are shared by both branches.
Chosen and rejected may contain independent multi-message assistant/tool
continuations, but every call must reference the shared inventory, every tool
response must resolve a call from its own branch, and both branches must end in
an assistant response. Invalid roles are rejected without compatibility repair.

`source_catalog.py` owns nine independently authored prompts per preference
dimension. It does not depend on SFT specs. The full 90-prompt catalog is
checked for semantic-source uniqueness, near-duplicates, template
concentration, and SFT prompt overlap before generation can construct a
provider backend. Requested finite source ranges are also validated. Model selection remains configurable through
`DPO_MODEL`; OpenRouter routing remains configurable through the shared routing
variables.
`DPO_ADJUDICATOR_MODEL` and `DPO_ADJUDICATOR_MAX_TOKENS` default to the renderer
settings and can be overridden. All three calls use the existing configurable
OpenRouter routing, retry, backoff, and adaptive request controls; manifests
retain aggregate and per-stage telemetry.

Finite briefs may declare internal machine-checkable output constraints. The
chosen branch must pass them before semantic adjudication. The rejected branch
is measured and recorded but may intentionally violate the requested
preference dimension and failure mode. Missing or failed chosen-branch evidence
blocks publication.

## Candidate Planning and Acceptance

Every selected preference dimension requires an explicit candidate count. Public output keeps the first pair for each normalized ID, prompt, and `(prompt, chosen, rejected)` triple. Duplicate or locally rejected candidates are not replaced merely to reach a nominal pair count.

Coverage reports include aggregate and per-dimension metadata, uniqueness, chosen/rejected similarity, negative-pattern distribution, holdout results, candidate/attempted/accepted/rejected/duplicate counts, and estimated accepted tokens. Similarity and repeated negative patterns are diagnostics; exact duplicates, holdout collisions, empty output, and stale accounting are publication blockers. Downstream repositories decide consumption.
Token estimates serialize the public prompt, chosen branch, rejected branch, and optional shared tools at four characters per token; the downstream tokenizer remains authoritative.

## Publication

`dpo-push` publishes one complete run to `DPO_HF_REPO` in one commit:

```text
README.md
data/<preference_dimension>.jsonl
artifacts/coverage.json
artifacts/manifests/*.manifest.json
```

The default dataset configuration loads all preference-dimension files. Named
configurations load one dimension without duplicating stored pairs.
Publication requires one flat final JSONL file per manifest dimension, exact
filename-to-metadata binding, and a current publish-ready report and card.
`generate-llm-run` is the only generic DPO generation command.

## Commands

Use `make dpo-smoke`, `make dpo-generate`, `make dpo-report`, `make dpo-inspect`, and `make dpo-push`. See `../../docs/COMMANDS.md` for variables and `../../docs/GENERATION_WORKFLOW.md` for the run ladder.
