# `configs`

## Purpose

Configuration inputs for pretraining generation and evaluation holdout protection.

This folder does not define SFT/DPO/distillation prompt semantics; those live in their owning dataset packages.

## Contents

~~~text
configs/
├── configure_synthetic.py    # renders the pretraining config from a profile
├── synthetic_template.yaml   # grounded pretraining config template
└── eval_holdouts.yaml        # registered evaluation holdout keys/prompts
~~~

`synthetic.yaml` is generated at runtime and is intentionally not a permanent source-of-truth file.

## Key Files

| File | Purpose |
|---|---|
| `configure_synthetic.py` | Writes the current pretraining run config with model, token target, batch size, concurrency, run id, and HF repo. |
| `synthetic_template.yaml` | Defines grounded signal mix and generation settings used by pretraining. |
| `eval_holdouts.yaml` | Prevents exact evaluation prompt/key leakage into supported alignment datasets. |

## How It Fits In

`make pretrain-smoke` and `make pretrain-generate` call `configure_synthetic.py` before running `slm_synth.pretrain.pipeline`.

Generic SFT, DPO, and Distillation DPO use the holdout registry during reporting/generation where configured.

See [Architecture](../docs/ARCHITECTURE.md).

## Usage

Generate a pretraining configuration directly:

~~~bash
python configs/configure_synthetic.py \
  --profile balanced \
  --tokens 100000 \
  --batch-size 32 \
  --concurrency 4 \
  --model openai/gpt-5.6-luna-pro \
  --run pretrain-smoke-001
~~~

Profiles affect default concurrency/max-token request posture; they do not restrict model choice or force sampling parameters, and they do not change the five pretraining signal definitions.

Supported profiles:

| Profile | Default concurrency | Sampling posture |
|---|---:|---|
| `speed` | 8 | higher default concurrency |
| `balanced` | 4 | default concurrency compromise |
| `quality` | 2 | lower default concurrency |

The Makefile sets role defaults, but every pretraining model role is replaceable. Suitability is determined by the live structured-output/reasoning-off contract, not a static model list.

## Conventions

- Do not hardcode provider routing into dataset configs; routing belongs in the shared runtime/Make variables.
- Do not add model-training hyperparameters here.
- Treat `target_total_tokens` as a **final accepted** pretraining target. The pipeline may generate more candidate tokens to compensate for validation/review/dedup losses.
- Holdout entries should represent real evaluation exclusions, not broad topic bans.
