# Architecture

How the five dataset products share execution mechanics while retaining dataset-specific quality semantics.

## System Boundary

This repository creates datasets. It does not train or evaluate models.

~~~text
OpenRouter / deterministic sources
              │
              ▼
      slm_synth/runtime
              │
   ┌──────────┼──────────┬──────────────────┬───────────────────┐
   ▼          ▼          ▼                  ▼                   ▼
pretrain     SFT        DPO        distillation SFT     distillation DPO
   │          │          │                  │                   │
   └──────────┴──────────┴──────────────────┴───────────────────┘
                              │
                       public JSONL
                       manifests/reports
                              │
                         Hugging Face
~~~

Downstream responsibilities:

- `slm` consumes pretraining/SFT/DPO datasets for model training.
- `slm-distillation` consumes distillation SFT and distillation DPO datasets.
- reasoning training, student sampling, logits, checkpoints, evaluation, and export are outside this repository.

## Shared Runtime

`slm_synth/runtime/` contains only mechanics that are semantically identical across dataset products.

| Module | Responsibility |
|---|---|
| `backend.py` | Build the OpenRouter backend with routing, provider, sampling, and adaptive request limits. |
| `batching.py` | Chunk work, split failed batches, and fill exact requested cardinality. |
| `stages.py` | Concurrent stage execution, bounded retry, recursive fault isolation, failure recording, and progress. |
| `novelty.py` | Normalization, shingling, exact matching, Jaccard/sequence near-duplicate checks. |
| `io.py` | JSON/JSONL write, append, and run-owned output reset helpers. |
| `reporting.py` | Shared deterministic-validation/quality evidence extraction and token estimates used by reports. |

The runtime does **not** define what a good SFT answer is, what makes a DPO preference valid, or which fields belong in a distillation row.

## Dataset Ownership

Each dataset package owns:

- seed/family/dimension semantics
- model prompts and strict output schemas
- deterministic row validation
- judge criteria
- reviewer criteria
- final acceptance and deduplication rules
- public metadata
- report/card format
- Hugging Face publication layout

This prevents a shared helper from silently weakening one dataset to match another.

## Production Stage Flows

### Pretraining

~~~text
deterministic grounded artifact
→ model rendering
→ deterministic record validation
→ Gemma semantic judge
→ Luna reviewer
→ final global exact dedup
→ post-review token accounting
→ backfill through the same full path if under target
→ deduped/pretrain.jsonl
~~~

Pretraining is token-targeted. Rows rejected by deterministic checks, judge, reviewer, or final dedup do not count toward the accepted token target.

### Generic SFT

~~~text
semantic derivation
→ concrete task
→ exact/near task novelty
→ answer
→ deterministic SFT schema/output checks
→ Nemotron judge
→ Gemma reviewer
→ final exact dedup
→ per-family public JSONL
~~~

The public product is one consolidated SFT repository. Families are organizational/configuration boundaries, not separate products.

### Generic DPO

~~~text
semantic derivation
→ concrete task
→ exact/near task novelty
→ chosen/rejected pair
→ deterministic pair validation
→ Nemotron judge
→ Gemma reviewer
→ final exact preference-triple dedup
→ per-dimension public JSONL
~~~

Generation uses plain-text prompt/chosen/rejected semantics internally. Final accepted rows are adapted to the public message-list schema.

### Distillation SFT

~~~text
semantic derivation
→ student-appropriate prompt
→ task novelty
→ teacher response
→ deterministic row/response validation
→ response novelty
→ Nemotron judge
→ Gemma reviewer
→ prompt/response exact dedup
→ per-signal public JSONL
~~~

This path applies stronger response-diversity controls than generic SFT because repeated teacher responses across unrelated prompts are a known distillation failure mode.

### Distillation DPO

~~~text
semantic derivation
→ concrete task
→ task novelty
→ chosen/rejected pair
→ deterministic pair validation
→ five-gate Gemma judge
→ Luna reviewer
→ final exact preference-triple dedup
→ teacher_response_preference.jsonl
~~~

The five judge gates are:

1. `assessable`
2. `chosen_complete`
3. `chosen_correct`
4. `preference_valid`
5. `dimension_aligned`

All five must pass.

## Quality Evidence

Model decisions are not treated as invisible transient state. Dataset pipelines persist enough evidence for reporting and publication checks to establish that:

- deterministic validation ran,
- judge decisions exist for assessable candidates,
- reviewer decisions exist for judge-accepted candidates,
- final published rows came from the accepted set,
- holdout constraints were checked where applicable.

Provider names, retry counts, cost, model lineage, and other execution metadata belong in manifests/reports rather than public training rows.

## Model Suitability

All five products share the same suitability policy:

- non-reasoning models may be eligible;
- reasoning-capable models are invoked with reasoning disabled when the provider/model supports it;
- a live structured request must succeed with reasoning disabled;
- models whose reasoning is mandatory or cannot be disabled are unsuitable.

Qualification uses the same strict structured-output backend shape used by production.

## Publication Boundary

Each product publishes to one consolidated Hugging Face repository.

| Product | Public organization |
|---|---|
| Pretraining | one `pretrain.jsonl` train artifact |
| SFT | family JSONL files in one repository |
| DPO | dimension JSONL files in one repository |
| Distillation SFT | signal JSONL files in one repository |
| Distillation DPO | one `teacher_response_preference.jsonl` file |

Internal work files, failed rows, raw provider responses, retry artifacts, and private lineage are not public training data.

## See Also

- [Generation Workflow](GENERATION_WORKFLOW.md)
- [Dataset Purpose and Contracts](DATASET_PURPOSE.md)
- [Command Reference](COMMANDS.md)
