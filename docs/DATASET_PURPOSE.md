# Dataset Purpose

This repository produces four dataset families. They are separate because each family has a different training objective and public row contract.

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

Teacher model, provider, generation run, signal, token target, difficulty, and internal metadata are excluded from public rows. Those details are stored in local manifests and dataset cards.

## SFT Data

SFT records are chat-style supervised examples.

Public SFT rows use this schema:

```json
{
  "id": "string",
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null"
  }
}
```

Teacher model, provider, generation run, retries, cost, task variables, and `holdout_key` are excluded from public rows. Run details stay in manifests.

## DPO Data

DPO records are preference examples with a preferred answer and a rejected answer.

Public DPO rows use this schema:

```json
{
  "id": "string",
  "prompt": [{"role": "user", "content": "string"}],
  "chosen": [{"role": "assistant", "content": "string"}],
  "rejected": [{"role": "assistant", "content": "string"}],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null",
    "failure_mode": "string"
  }
}
```

Teacher model, provider, generation run, retries, cost, task variables, and `holdout_key` are excluded from public rows. Run details stay in manifests.

## Taxonomy

| Field | Meaning | Used by |
|---|---|---|
| `category` | Training objective | SFT, DPO |
| `eval_family` | Eval-shaped behavior pattern | SFT, DPO, holdout checks |
| `template_family` | Generation/template surface | SFT, DPO |
| `failure_mode` | Rejected-answer behavior | DPO only |
| `holdout_key` | Exact local structured holdout guard | Spec validation and materialization |

Same-family training examples are allowed when variables and templates differ from eval items. Exact eval prompts and matching structured holdout keys are rejected.

## Token Targets

Distillation token targets are planning presets:

| Preset | Total tokens |
|---|---:|
| `smoke` | 100K |
| `pilot` | 1M |
| `scale-check` | 10M |
| `final` | 100M |

The planner converts token targets into approximate row counts using an estimated tokens-per-row value. Final training-token counts should be measured downstream with the actual training tokenizer.
