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

All signals are exported together in one `data/pretrain.jsonl` dataset file:

```json
{
  "id": "pretrain_<content hash>",
  "text": "string",
  "metadata": {"signal": "string"}
}
```

Validated structured rows and per-signal files are internal generation
artifacts. Public acceptance applies exact and structural near-duplicate checks
globally, including across signals. Rejected candidates do not trigger quota
backfill, and publication is blocked if the consolidated file fails the same
full-file uniqueness audit.

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
    {"role": "system", "content": "string"},
    {"role": "user", "content": "string"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": {"query": "string"}}
      }]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "lookup",
      "description": "string",
      "parameters": {"type": "object", "properties": {}}
    }
  }],
  "metadata": {
    "task_family": "string",
    "interaction_modes": ["string"],
    "output_mode": "string",
    "context_mode": "string",
    "difficulty": 1,
    "template_family": "string"
  }
}
```

`system` and `tools` are optional. Tool-mediated rows require declared tools,
structured calls, matching tool responses, and a final assistant message.
Ordinary and multi-turn rows use strict user/assistant alternation; malformed
adjacent roles are rejected rather than merged. Task variables, `holdout_key`,
teacher/provider details, retries, and cost stay out of public rows.

## DPO Data

Public DPO rows use this schema:

```json
{
  "id": "string",
  "prompt": [{"role": "user", "content": "string"}],
  "chosen": [
    {"role": "assistant", "content": null, "tool_calls": [{
      "id": "call_1", "type": "function",
      "function": {"name": "lookup", "arguments": {"query": "correct query"}}
    }]},
    {"role": "tool", "tool_call_id": "call_1", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "rejected": [
    {"role": "assistant", "content": null, "tool_calls": [{
      "id": "call_2", "type": "function",
      "function": {"name": "lookup", "arguments": {"query": "wrong query"}}
    }]},
    {"role": "tool", "tool_call_id": "call_2", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "lookup", "description": "string",
      "parameters": {"type": "object", "properties": {}}
    }
  }],
  "metadata": {
    "task_family": "string",
    "interaction_modes": ["string"],
    "output_mode": "string",
    "context_mode": "string",
    "difficulty": 1,
    "template_family": "string",
    "preference_dimension": "string",
    "failure_mode": "string"
  }
}
```

`prompt` and `tools` occur once and are shared by both branches. Chosen and
rejected may each contain a complete multi-message tool-use continuation, but
may not embed a different prompt or tool inventory. Calls in both branches must
reference the shared tools and resolve their own call IDs. Chosen and rejected
responses are public. Teacher/provider details, retries, cost, task variables,
and `holdout_key` stay in manifests.

Published generic DPO data uses one repository containing one
`data/<preference_dimension>.jsonl` file per preference dimension. The default
dataset configuration loads every dimension file; named configurations select
one dimension. These configurations do not create train/validation/test splits
or duplicate stored pairs.

## Distillation SFT Data

Distillation SFT rows are teacher prompt/response examples. Public artifacts are per-signal JSONL files; downstream train/validation/test splitting belongs to `slm-distillation`.

Public distillation SFT rows use this schema:

```json
{
  "id": "string",
  "prompt": "string",
  "reasoning": null,
  "response": "string",
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null"
  }
}
```

`reasoning` is always null in public rows. Public metadata supports record-level filtering and audit. Signal, prompt-source, provider, teacher model, routing, retries, cost, planning, and response-gate details stay in manifests and dataset cards.

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
| `task_family` | Broad task objective. | Generic SFT and DPO |
| `interaction_modes` | Conversation shape: single-turn, multi-turn, system-conditioned, or tool-mediated. | Generic SFT and DPO |
| `output_mode` | Required response surface. | Generic SFT and DPO |
| `context_mode` | Source-context shape. | Generic SFT and DPO |
| `preference_dimension` | Quality dimension separating chosen from rejected. | Generic DPO |
| `category` | Distillation training objective. | Distillation SFT and DPO |
| `eval_family` | Distillation-only behavior label pending the separate distillation review. | Distillation SFT and DPO |
| `template_family` | Generation/template surface. | SFT, DPO, distillation SFT, distillation DPO |
| `failure_mode` | Rejected-answer behavior. | DPO, distillation DPO |
| `holdout_key` | Exact structured holdout guard. | Spec validation and materialization |

Generic SFT/DPO rows do not expose evaluation-family labels. Exact normalized prompt fingerprints and matching structured `holdout_key` values are rejected locally; both controls remain internal and are never sent to teachers or published.

Generic SFT publication additionally requires a clean accepted-row audit:
there may be no exact or near-duplicate prompts/conversations, repeated
assistant-response clusters, concentrated templates, invalid role/tool
sequences, failed or missing semantic-adjudication evidence, or holdout
collisions. Rejected candidates remain recorded outcomes; adjudication evidence
is required for rows that survive into the public files.

## Generation Budgets

Generic SFT, generic DPO, and distillation SFT use explicit candidate counts by
task family, preference dimension, or signal. Candidate counts limit provider work; they are not accepted-output quotas.
Quality rejections and duplicates reduce the published row count and are not
backfilled merely to reach a requested size.

The generated corpus reports candidate, attempted, accepted, rejected,
duplicate, and token counts. Dataset selection, token budgets, mixtures,
sequence lengths, epochs, and model-size-specific consumption belong to the
downstream training repositories.

Generic DPO does not backfill rejected candidates to reach a nominal pair count.
Distillation planning remains separate pending its dedicated review.

## See Also

- `GENERATION_WORKFLOW.md` for the end-to-end run ladder.
- `GENERATION_FAMILIES.md` for supported families/signals and target distribution behavior.
- `COMMANDS.md` for Make targets and common variables.
- `../slm_synth/README.md` for package layout.
