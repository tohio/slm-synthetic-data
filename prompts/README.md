# Prompts

This directory contains prompt helpers for the pretraining synthetic workflow.

The current production pretraining path uses grounded artifacts plus OpenRouter structured rendering. Distillation prompt records live under `slm_synth/distillation/` because response distillation has a different public schema and teacher contract.

## Pretraining Contract

Pretraining renderer prompts ask the model to return a JSON object with an `items` array:

```json
{
  "items": [
    { "...": "..." }
  ]
}
```

Prompts should:

- keep the top-level JSON object contract,
- avoid markdown and code fences,
- keep string fields JSON-safe,
- preserve the schema expected by `slm_synth.pretrain.schemas`,
- avoid generic repeated examples,
- keep final records concise.

## Pretraining Signals

| Signal | Purpose |
|---|---|
| `arithmetic` | Render verified integer arithmetic artifacts into question, steps, and answer records. |
| `task_code` | Render local Python task artifacts into task, plan, and function records. |
| `educational_qa_mcq_math` | Render verified math MCQ artifacts into question, choices, answer index, and explanation records. |
| `educational_qa_mcq_general` | Render evidence-grounded MCQ artifacts into question, choices, answer index, and explanation records. |
| `factual_restraint` | Render controlled uncertainty/privacy/context artifacts into cautious answer records. |

## Smoke Check

After changing pretraining prompts or prompt wrappers:

```bash
make configure TOKENS=100000 BATCH=32 CONCURRENCY=4 RUN=prompt_smoke
make preflight-artifacts
make generate
make validate
make dedup
make report-duplicates STAGE=deduped
```
