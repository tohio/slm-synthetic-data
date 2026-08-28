# Dataset Purpose and Contracts

What each of the five public dataset products is intended to teach, how it is represented, and where it is consumed.

## Product Summary

| Product | Primary purpose | Public organization | Primary consumer |
|---|---|---|---|
| Pretraining | Broad synthetic language/code/reasoning continuation data grounded in deterministic artifacts. | one consolidated `pretrain.jsonl` | `slm` pretraining/continued pretraining |
| Generic SFT | Teach instruction following, interaction behavior, task execution, code, reasoning, and safe uncertainty. | one repo with family JSONL files | `slm` SFT |
| Generic DPO | Teach semantic preference between stronger and weaker responses. | one repo with dimension JSONL files | `slm` DPO |
| Distillation SFT | Transfer high-quality teacher responses to a student model. | one repo with signal JSONL files | `slm-distillation` response distillation |
| Distillation DPO | Teach post-distillation preference behavior. | one consolidated `teacher_response_preference.jsonl` | `slm-distillation` DPO |

The five products are separate because their training consumers, public schemas, and acceptance semantics differ.

## Public Row Contracts

### Pretraining

Pretraining rows are signal-specific JSON objects rather than one chat schema. The five supported record types are:

- `arithmetic`
- `task_code`
- `educational_qa_mcq_math`
- `educational_qa_mcq_general`
- `factual_restraint`

Examples of training-bearing fields include:

| Signal | Core content |
|---|---|
| arithmetic | question, steps, answer |
| task_code | task, plan, code |
| math MCQ | question, choices, explanation, answer index |
| general MCQ | question, choices, explanation, answer index |
| factual restraint | question, safe answer |

All signals are combined into one final `deduped/pretrain.jsonl`.

### Generic SFT

Required public fields:

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

Some interaction modes may include system/tool structures allowed by the SFT schema. Public metadata records taxonomy and interaction labels, not provider/run telemetry.

### Generic DPO

Required public fields:

~~~json
{
  "id": "...",
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {}
}
~~~

Internally, model-facing generation uses plain prompt/chosen/rejected text; the pipeline adapts accepted rows to the public message schema only after quality acceptance.

### Distillation SFT

Required public fields:

~~~json
{
  "id": "...",
  "prompt": "...",
  "reasoning": null,
  "response": "...",
  "metadata": {}
}
~~~

`reasoning` is deliberately fixed to `null`. Teacher/provider lineage, retries, cost, and private generation metadata belong in manifests, not training rows.

### Distillation DPO

Required public fields:

~~~json
{
  "id": "...",
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {}
}
~~~

Public metadata requires a failure-mode label because the dataset is explicitly preference-oriented.

## Acceptance Boundaries

### Pretraining

A row counts toward the target only after:

1. grounded generation;
2. deterministic validation;
3. semantic judge acceptance;
4. reviewer acceptance;
5. final global exact dedup.

Accepted-token accounting is performed after all five steps.

### SFT

Final accepted rows must survive:

1. task novelty;
2. answer generation;
3. deterministic schema/output-constraint validation;
4. judge;
5. reviewer;
6. final exact dedup.

### Generic DPO

A pair must have distinct chosen/rejected branches, satisfy the public pair contract, survive semantic judge/reviewer checks, and remain unique after final triple dedup.

### Distillation SFT

In addition to row validity, the response itself must be sufficiently novel. Final acceptance rejects:

- duplicate prompt+response pairs;
- duplicate prompts;
- duplicate responses.

This prevents a teacher from producing a generic repeated answer across unrelated student prompts.

### Distillation DPO

The judge requires all five semantic gates before review:

- assessable;
- chosen complete;
- chosen correct;
- preference valid;
- dimension aligned.

This is intentionally stricter than generic DPO.

## Metadata and Private Execution Data

Public metadata may contain taxonomy information such as:

- category
- difficulty
- template family
- eval family
- failure mode
- interaction mode

The following belong in manifests/reports instead of training rows:

- provider
- teacher/judge/reviewer model ids where not explicitly part of a public card
- retry count
- routing decisions
- request latency
- request token counts
- cost
- batch ids
- internal prompt/spec structures
- stage failure details

## Holdout Policy

Generic SFT, generic DPO, and Distillation DPO use the configured evaluation holdout registry where appropriate. Exact evaluation prompts and matching holdout keys must not leak into training data.

Same-family training examples are allowed when they differ in task variables/content and do not collide with registered holdout keys.

## Downstream Boundary

This repository does not decide:

- training epochs;
- learning rates;
- model-size-specific data mixture;
- tokenizer changes;
- checkpoint policy;
- student sampling;
- logit distillation;
- evaluation suites;
- model export.

Those decisions belong to the repositories that consume these datasets.

## See Also

- [Generation Families and Dimensions](GENERATION_FAMILIES.md)
- [Architecture](ARCHITECTURE.md)
- [Generation Workflow](GENERATION_WORKFLOW.md)
