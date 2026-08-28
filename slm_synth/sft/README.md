# `slm_synth/sft`

## Purpose

Generate the generic supervised fine-tuning dataset used to teach broad instruction-following behavior.

The package owns SFT family semantics, task specifications, answer validation, judge/reviewer quality, public schema, reports, and consolidated publication. It does not generate DPO pairs or distillation-specific rows.

## Contents

~~~text
sft/
├── pipeline.py             # supported derivation → acceptance production path
├── spec_builders.py        # family specification/source construction
├── specs.py                # spec validation / teacher-visible shaping
├── source_catalog.py       # source inventory for applicable families
├── planning.py             # SFT-local source planning helpers
├── schema.py               # public SFT message schema
├── acceptance.py           # normalized final uniqueness
├── publication_quality.py  # publication evidence checks
├── report.py               # coverage/acceptance reporting
├── card.py                 # consolidated dataset-card configuration
├── push_hf.py              # atomic consolidated HF publication
└── cli.py                  # report CLI
~~~

## Production Flow

~~~text
semantic derivation
→ concrete task
→ exact/near task novelty
→ answer generation
→ deterministic schema/output validation
→ Nemotron judge
→ Gemma reviewer
→ final exact dedup
→ datasets/<family>.jsonl
~~~

Default model roles:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| answer | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

## How It Fits In

The pipeline uses `slm_synth/runtime` for execution mechanics while this package owns all SFT semantics.

See [Architecture](../../docs/ARCHITECTURE.md) and [Generation Families](../../docs/GENERATION_FAMILIES.md).
For every supported override, see [SFT parameters](../../docs/PARAMETERS.md#generic-sft-parameters).

## Usage

~~~bash
make sft-smoke

SFT_GENERATION_RUN=sft-production-001 make sft-generate
SFT_INSPECT_RUN=sft-production-001 make sft-inspect
SFT_REPORT_RUN=sft-production-001 make sft-report
SFT_PUSH_RUN=sft-production-001 make sft-push
~~~

Use a single family for focused smoke work:

~~~bash
SFT_FAMILIES=programming make sft-smoke
~~~

## Public Contract

Minimum public row:

~~~json
{
  "id": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {}
}
~~~

The schema also supports interaction structures required by system-conditioned, multi-turn, or tool-mediated families.

## Outputs

~~~text
data/sft/runs/<run>/
├── datasets/<family>.jsonl
├── manifests/
├── work/<family>/
├── coverage.json
└── README.md
~~~

Only `datasets/` is public training data. `work/` holds derivations, tasks, answer candidates, decisions, rejections, failures, and summaries.

## Conventions

- Families are internal coverage/configuration boundaries inside one SFT product.
- Novelty filtering occurs before answer generation.
- Deterministic validation occurs before judge.
- Reviewer sees judge-accepted rows only.
- Provider/run/cost/retry details stay out of public rows.
- Holdout registry checks must not be bypassed.

## Gotchas

A high acceptance rate is not itself a goal. The quality system should reject malformed, ungrounded, incorrect, repetitive, or instruction-violating examples rather than optimizing for 100% yield.
