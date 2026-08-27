# `slm_synth`

`slm_synth` contains five dataset packages and one shared runtime.

```text
slm_synth/
├── runtime/              # shared mechanical execution only
├── pretrain/             # consolidated synthetic pretraining dataset
├── sft/                  # generic SFT dataset
├── dpo/                  # generic DPO dataset
├── distillation_sft/     # teacher response distillation dataset
├── distillation_dpo/     # post-distillation preference dataset
├── taxonomy/             # shared public metadata labels/holdouts
├── hf_push.py            # shared HF commit/readiness mechanics
├── llm.py                # OpenRouter client/retries/routing
└── qualify_model.py      # five-dataset model suitability checks
```

## Runtime Boundary

`runtime/` contains only mechanics reused across dataset products: backend construction, batching/cardinality, concurrent stage execution and isolation, novelty filtering, JSON/JSONL IO, and shared reporting mechanics.

Dataset packages retain all semantic decisions: prompts, schemas, deterministic checks, judge/reviewer criteria, final dedup rules, reports, and publication presentation.

## Supported Production Paths

Each dataset has one pipeline module and one public Make workflow. Legacy generation/orchestration compatibility paths have been removed.
