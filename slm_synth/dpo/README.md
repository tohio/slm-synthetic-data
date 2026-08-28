# `slm_synth/dpo`

## Purpose

Generate the generic preference dataset used for DPO alignment.

The package owns preference-dimension semantics, pair generation, deterministic pair validation, judge/reviewer acceptance, public schema, reports, and consolidated publication. It remains semantically separate from Distillation DPO.

## Contents

~~~text
dpo/
├── pipeline.py       # supported derivation → pair → acceptance path
├── schema.py         # public prompt/chosen/rejected message schema
├── acceptance.py     # final normalized uniqueness
├── report.py         # coverage/quality/holdout reporting
├── card.py           # consolidated dataset-card configurations
├── push_hf.py        # atomic consolidated HF publication
├── io.py             # DPO-local file helpers
├── cli.py            # report CLI
└── __init__.py
~~~

## Production Flow

~~~text
semantic derivation
→ concrete task
→ exact/near task novelty
→ chosen/rejected pair generation
→ deterministic pair validation
→ Nemotron judge
→ Gemma reviewer
→ exact prompt/chosen/rejected dedup
→ datasets/<dimension>.jsonl
~~~

Default model roles:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| pair | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

## How It Fits In

Internal model stages use plain text because preference semantics are easiest to adjudicate directly. Accepted rows are adapted to the public message-list DPO schema before publication.

See [Architecture](../../docs/ARCHITECTURE.md) and [Generation Families](../../docs/GENERATION_FAMILIES.md).
For every supported override, see [DPO parameters](../../docs/PARAMETERS.md#generic-dpo-parameters).

## Usage

~~~bash
make dpo-smoke

DPO_GENERATION_RUN=dpo-production-001 make dpo-generate
DPO_INSPECT_RUN=dpo-production-001 make dpo-inspect
DPO_REPORT_RUN=dpo-production-001 make dpo-report
DPO_PUSH_RUN=dpo-production-001 make dpo-push
~~~

Focused smoke:

~~~bash
DPO_PREFERENCE_DIMENSIONS=code_correctness make dpo-smoke
~~~

## Public Contract

~~~json
{
  "id": "...",
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {}
}
~~~

## Outputs

~~~text
data/dpo/runs/<run>/
├── datasets/<dimension>.jsonl
├── manifests/
├── work/<dimension>/
├── coverage.json
└── README.md
~~~

## Conventions

- Chosen and rejected must differ materially, not cosmetically.
- Pair validity is checked deterministically before model adjudication.
- Judge/reviewer criteria are generic-DPO criteria; do not import Distillation-DPO five-gate semantics here.
- Final uniqueness covers prompt and preference triple behavior as implemented by the pipeline.
- Evaluation holdout checks must remain enabled for production reports/publication.
