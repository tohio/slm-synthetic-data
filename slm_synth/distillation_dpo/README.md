# `slm_synth/distillation_dpo`

## Purpose

Generate preference pairs for post-distillation DPO alignment.

Distillation DPO is intentionally stricter than generic DPO. It uses distillation-specific derivation guidance, controlled weak-response defects, a five-gate judge, and independent reviewer calibration.

## Contents

~~~text
distillation_dpo/
├── pipeline.py        # supported production path
├── seeds.py           # public family metadata / validator helpers
├── schema.py          # public pair schema
├── pair_quality.py    # deterministic pair-quality checks
├── acceptance.py      # dataset-level acceptance/readiness
├── io.py              # dataset IO
├── report.py          # coverage/holdout reporting
├── card.py            # dataset-card generation
├── push_hf.py         # consolidated HF publication
└── cli.py             # report/card CLI
~~~

## Production Flow

~~~text
semantic derivation
→ concrete task
→ task novelty
→ chosen/rejected pair
→ deterministic pair validation
→ five-gate Gemma judge
→ Luna reviewer
→ final exact triple dedup
→ datasets/teacher_response_preference.jsonl
~~~

Default models:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| pair | `deepseek/deepseek-v4-flash` |
| judge | `google/gemma-4-31b-it` |
| reviewer | `openai/gpt-5.6-luna-pro` |

## Five-Gate Judge

A pair is accepted by the judge only if all are true:

1. `assessable`
2. `chosen_complete`
3. `chosen_correct`
4. `preference_valid`
5. `dimension_aligned`

Reviewer sees only judge-accepted pairs and independently decides whether the acceptance was correct.

## How It Fits In

The public dataset is consumed by `slm-distillation` after response distillation when DPO alignment is required.

Do not merge its semantics with generic DPO merely because both use the same ten dimension names.

See [Architecture](../../docs/ARCHITECTURE.md).

## Usage

~~~bash
make distillation-dpo-smoke

DISTILLATION_DPO_TARGET_RUN=distillation-dpo-production-001 \
make distillation-dpo-generate

DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-production-001 \
make distillation-dpo-inspect

DISTILLATION_DPO_REPORT_RUN=distillation-dpo-production-001 \
make distillation-dpo-report

DISTILLATION_DPO_PUSH_RUN=distillation-dpo-production-001 \
make distillation-dpo-push
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
data/distillation-dpo/runs/<run>/
├── datasets/teacher_response_preference.jsonl
├── manifests/
├── work/<dimension>/
├── coverage.json
└── README.md
~~~

All selected dimensions contribute to the same consolidated public file.

## Conventions

- The rejected branch should contain a controlled defect relevant to the selected preference dimension.
- Chosen must be independently correct and complete; a bad rejected answer does not excuse a bad chosen answer.
- The pair must exhibit a material preference margin.
- Holdout checks remain enabled.
- Provider/run/retry/cost information stays in manifests rather than public rows.
