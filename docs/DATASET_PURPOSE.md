# Dataset Purpose

The repository publishes five distinct dataset products. Internal families/signals are components of those products, not separate repository products.

| Dataset | Purpose | Primary public output | Downstream consumer |
|---|---|---|---|
| Pretraining | Synthetic text for continued/base-model pretraining mixtures. | One consolidated `pretrain.jsonl` | `slm` pretraining/curation workflows |
| Generic SFT | Instruction-following supervised fine-tuning data. | Consolidated SFT repository with family JSONL files/configurations | `slm` SFT |
| Generic DPO | Preference pairs for generic alignment. | Consolidated DPO repository with preference-dimension files/configurations | `slm` DPO |
| Distillation SFT | Teacher prompt/response data for response distillation. | Consolidated distillation-SFT repository | `slm-distillation` response distillation |
| Distillation DPO | Preference pairs for post-distillation alignment. | Consolidated `teacher_response_preference.jsonl` repository | `slm-distillation` DPO |

## Shared Rules

- Provider, retry, cost, routing, and model lineage belong in manifests, not public rows.
- Deterministic validation occurs before model-based acceptance.
- Judge/reviewer evidence is persisted for quality-controlled paths.
- Exact/near-duplicate handling is dataset-specific and occurs before final publication.
- Evaluation holdout collisions are rejected on dataset paths that use the holdout registry.
- Model-size-specific data selection and training budgets belong in downstream training repositories.

## Pretraining Signals

The five pretraining signals are internal components of one pretraining dataset:

- `arithmetic`
- `task_code`
- `educational_qa_mcq_math`
- `educational_qa_mcq_general`
- `factual_restraint`

They are globally deduplicated and published together.

## Generic SFT and DPO Taxonomy

Generic SFT uses broad task families. Generic DPO uses preference dimensions. These labels remain metadata/configuration boundaries inside one SFT product and one DPO product respectively; they are not train/validation/test splits.

## Distillation Boundary

Distillation SFT and Distillation DPO remain separate products because they serve different training stages and public schemas. This repository generates their teacher-derived datasets only; student sampling, training, logits, checkpoints, and model export are outside this repository.
