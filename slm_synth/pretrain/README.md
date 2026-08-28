# `slm_synth/pretrain`

## Purpose

Build one consolidated synthetic pretraining dataset from five deterministic grounded signal families.

This package owns grounded artifact rendering, deterministic validation, semantic quality review, final global dedup, accepted-token completion, reporting, and pretraining publication. It does not create SFT conversations or preference pairs.

## Contents

~~~text
pretrain/
├── artifacts/              # deterministic grounded source objects
├── pipeline.py             # supported end-to-end production controller
├── curate.py               # internal grounded generation/validation/backfill engine
├── generate.py             # grounded provider rendering
├── grounded.py             # artifact → provider prompt/render persistence
├── quality.py              # Gemma judge + Luna reviewer semantic quality
├── record_quality.py       # deterministic record validators / canonical keys
├── validate.py             # deterministic raw → validated implementation
├── dedup.py                # internal exact dedup implementation
├── manifest.py             # run manifest aggregation
├── preflight_artifacts.py  # deterministic artifact quality preflight
├── report_artifacts.py     # artifact coverage/quality report
├── report_diversity.py     # final structural-diversity audit
├── schemas.py              # signal schemas
└── push_hf.py              # consolidated HF publication
~~~

## Key Files

| File | Purpose |
|---|---|
| `pipeline.py` | Owns the production quality loop and post-review accepted-token completion. |
| `curate.py` | Generates enough grounded candidates to satisfy a candidate token target; called internally by `pipeline.py`. |
| `quality.py` | Applies signal-specific semantic judge criteria and reviewer confirmation. |
| `record_quality.py` | Deterministic correctness/schema checks and canonical exact keys. |
| `push_hf.py` | Verifies semantic completion and uploads only the final consolidated dataset. |

## How It Fits In

Production flow:

~~~text
deterministic artifact
→ model rendering
→ deterministic validation
→ Gemma judge
→ Luna reviewer
→ global exact dedup
→ accepted-token calculation
→ full-path backfill if required
→ deduped/pretrain.jsonl
~~~

See [Architecture](../../docs/ARCHITECTURE.md).
For every supported override, see [Pretraining parameters](../../docs/PARAMETERS.md#pretraining-parameters).

## Usage

~~~bash
make pretrain-smoke

PRETRAIN_TARGET_RUN=pretrain-production-001 \
PRETRAIN_TARGET_TOKENS=1000000 \
make pretrain-generate

PRETRAIN_INSPECT_RUN=pretrain-production-001 make pretrain-inspect
PRETRAIN_REPORT_RUN=pretrain-production-001 make pretrain-report
~~~

## Outputs

Primary public artifact:

~~~text
data/runs/<run>/deduped/pretrain.jsonl
~~~

Completion evidence:

~~~text
data/runs/<run>/manifests/accepted_token_report.json
~~~

The five internal signals are:

~~~text
arithmetic
task_code
educational_qa_mcq_math
educational_qa_mcq_general
factual_restraint
~~~

They are not published as separate repositories.

## Conventions

- Artifacts must be deterministic and locally inspectable before provider rendering.
- Accepted tokens are counted only after deterministic validation, judge, reviewer, and final dedup.
- Backfill must pass through the complete quality path.
- Reviewer sampling is not forced to zero for Luna; provider-compatible defaults are preserved.
- A single-signal run is a debugging/diagnostic option, not a different public product.

## Gotchas

`curate.py`, `validate.py`, and `dedup.py` remain implementation details because the integrated pipeline still calls their logic. They are not separate supported user workflows.
