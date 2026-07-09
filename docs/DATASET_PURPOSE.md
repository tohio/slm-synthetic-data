# Dataset Purpose

Reference for artifact families, public row contracts, and metadata boundaries.

## Artifact Families

| Family | Purpose | Primary consumer |
|---|---|---|
| Pretraining synthetic data | Grounded text signals for pretraining or continued-pretraining mixes. | `slm` pretraining curation |
| `sft-*` | Generic chat-style supervised fine-tuning examples. | `slm` SFT |
| `dpo-*` | Generic preference pairs for alignment. | `slm` DPO |
| `distillation-sft-*` | Teacher prompt/response rows for response distillation. | `slm-distillation` SFT |
| `distillation-dpo-*` | Preference pairs for aligning distilled models. | `slm-distillation` DPO |

Provider, teacher, run, retry, cost, routing, and internal prompt-spec details belong in manifests and dataset cards, not public rows.

## Pretraining Synthetic Data

Pretraining records are targeted synthetic text signals for mixing into a broader raw or curated pretraining corpus. They are not SFT, DPO, or distillation rows.

| Signal | Purpose |
|---|---|
| `arithmetic` | Numeric reasoning coverage with verified integer arithmetic. |
| `task_code` | Python code-pattern exposure from local task specifications. |
| `educational_qa_mcq_math` | Mathematical multiple-choice discrimination with verification. |
| `educational_qa_mcq_general` | Educational multiple-choice discrimination grounded in supplied evidence. |
| `factual_restraint` | Cautious-answer behavior for uncertainty, privacy, and missing-context cases. |

## SFT Data

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

Task variables, `holdout_key`, teacher/provider details, retries, and cost stay out of public rows.

## DPO Data

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

Chosen and rejected responses are public. Teacher/provider details, retries, cost, task variables, and `holdout_key` stay in manifests.

## Distillation SFT Data

Distillation SFT rows are teacher prompt/response examples. Public artifacts are per-signal JSONL files; downstream train/validation/test splitting belongs to `slm-distillation`.

Public distillation SFT rows use this schema:

```json
{"id": "string", "prompt": "string", "reasoning": null, "response": "string"}
```

`reasoning` is always null in public rows. Signal, prompt-source, provider, teacher model, routing, retries, cost, planning, and response-gate details stay in manifests and dataset cards.

## Distillation DPO Data

Distillation DPO rows are preference pairs for aligning distilled models. They are isolated from generic `dpo-*` artifacts.

Public distillation DPO rows use this schema:

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

Production distillation DPO pairs use teacher-quality chosen responses and controlled-weak rejected responses. Student-model sampling is not part of this repository.

## Taxonomy

| Field | Meaning | Used by |
|---|---|---|
| `category` | Training objective. | SFT, DPO, distillation DPO |
| `eval_family` | Eval-shaped behavior pattern. | SFT, DPO, holdout checks |
| `template_family` | Generation/template surface. | SFT, DPO, distillation DPO |
| `failure_mode` | Rejected-answer behavior. | DPO, distillation DPO |
| `holdout_key` | Exact structured holdout guard. | Spec validation and materialization |

Same-family training examples are allowed when variables and templates differ from eval items. Exact eval prompts and matching structured holdout keys are rejected.

## Planning Knobs

| Surface | Target knob | Default |
|---|---|---:|
| Pretraining | `PRETRAIN_TARGET_TOKENS` | `1000000` |
| SFT | `SFT_TARGET_ROWS` | `14000` |
| DPO | `DPO_TARGET_PAIRS` | `14000` |
| Distillation SFT | `DISTILLATION_SFT_TARGET_ROWS` | `100000` |
| Distillation DPO | `DISTILLATION_DPO_TARGET_PAIRS` | `50000` |

SFT and DPO also keep `SFT_COUNT_PER_FAMILY` and `DPO_COUNT_PER_FAMILY` as explicit lower-level overrides. Production distillation planning uses accepted row/pair counts.

## See Also

- `GENERATION_WORKFLOW.md` for the end-to-end run ladder.
- `COMMANDS.md` for Make targets and common variables.
- `../slm_synth/README.md` for package layout.
