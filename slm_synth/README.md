# `slm_synth`

## Purpose

Python package for the five synthetic dataset products and their shared execution/runtime infrastructure.

It owns dataset generation and publication mechanics. It does not train models.

## Contents

~~~text
slm_synth/
├── runtime/              # shared mechanical execution primitives
├── pretrain/             # synthetic pretraining product
├── sft/                  # generic SFT product
├── dpo/                  # generic DPO product
├── distillation_sft/     # teacher-response distillation product
├── distillation_dpo/     # post-distillation DPO product
├── taxonomy/             # shared public metadata / holdout labels
├── llm.py                # OpenRouter client, routing, retry, telemetry
├── model_suitability.py  # reasoning/suitability policy
├── qualify_model.py      # live role qualification
├── hf_push.py            # shared atomic HF publication helpers
├── cards.py              # dataset-card generation
└── manifest_totals.py    # public manifest count normalization
~~~

## Key Files

| File | Purpose |
|---|---|
| `llm.py` | Provider-facing OpenRouter implementation used by the shared runtime backend. |
| `model_suitability.py` | Enforces reasoning-disable requirements and mandatory-reasoning rejection. |
| `qualify_model.py` | Verifies models against the same strict structured-output contract used by production. |
| `hf_push.py` | Shared commit/readiness mechanics; dataset packages still own publication semantics. |

## How It Fits In

See [Architecture](../docs/ARCHITECTURE.md).

The design boundary is deliberate:

- `runtime/` shares mechanics;
- dataset packages own semantics;
- top-level helpers exist only when they are genuinely cross-dataset.

## Conventions

- Do not put dataset-specific prompts or quality criteria in `runtime/`.
- Do not create a second generation path beside a dataset's `pipeline.py`.
- Public training rows must not contain provider/cost/retry/internal lineage.
- New model roles must use the suitability policy rather than bypass it with direct provider code.
