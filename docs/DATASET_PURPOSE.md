# Dataset Purpose

This repository produces two different dataset families. They are intentionally separate.

## Pretraining Synthetic Data

Pretraining records are targeted synthetic signals for a broader pretraining or continued-pretraining mix.

| Signal | Purpose |
|---|---|
| `arithmetic` | Numeric reasoning coverage with verified integer arithmetic. |
| `task_code` | Python code-pattern exposure from local task specifications. |
| `educational_qa_mcq_math` | Mathematical multiple-choice discrimination with verification. |
| `educational_qa_mcq_general` | Educational multiple-choice discrimination grounded in supplied evidence. |
| `factual_restraint` | Cautious-answer behavior for uncertainty, privacy, and missing-context cases. |

Pretraining records are not SFT, DPO, or response-distillation rows. They are intended to be mixed with broader raw or curated pretraining data downstream.

## Response Distillation Data

Distillation records are prompt-response examples generated from local prompts and teacher responses.

Public distillation rows use this schema:

```json
{"id": "string", "prompt": "string", "reasoning": null, "response": "string"}
```

`reasoning` may also be a list of strings when step-by-step supervision is useful.

Teacher model, provider, generation run, signal, difficulty, and internal metadata are excluded from public rows. Those details are stored in local manifests and dataset cards.

## Token Targets

Distillation token targets are planning presets:

| Preset | Total tokens |
|---|---:|
| `smoke` | 100K |
| `pilot` | 1M |
| `scale-check` | 10M |
| `final` | 100M |

The planner converts token targets into approximate row counts using an estimated tokens-per-row value. Final training-token counts should be measured downstream with the actual training tokenizer.
