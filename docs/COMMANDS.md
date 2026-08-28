# Command Reference

The Makefile intentionally exposes one production path per dataset.

## Dataset Targets

| Dataset | Smoke | Generate | Inspect | Report | Push |
|---|---|---|---|---|---|
| Pretrain | `pretrain-smoke` | `pretrain-generate` | `pretrain-inspect` | `pretrain-report` | `pretrain-push` |
| SFT | `sft-smoke` | `sft-generate` | `sft-inspect` | `sft-report` | `sft-push` |
| DPO | `dpo-smoke` | `dpo-generate` | `dpo-inspect` | `dpo-report` | `dpo-push` |
| Distillation SFT | `distillation-sft-smoke` | `distillation-sft-generate` | `distillation-sft-inspect` | `distillation-sft-report` | `distillation-sft-push` |
| Distillation DPO | `distillation-dpo-smoke` | `distillation-dpo-generate` | `distillation-dpo-inspect` | `distillation-dpo-report` | `distillation-dpo-push` |

Run `make help` for the same matrix.

## Shared Variables

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON` | `python` | Python executable |
| `MODEL` | `deepseek/deepseek-v4-flash` | Shared fallback model variable |
| `MAX_TOKENS` | `4096` | Shared token default |
| `OPENROUTER_ROUTING_MODE` | `auto` | `auto`, `prefer`, or `strict` |
| `OPENROUTER_PROVIDER` | unset | Provider for `prefer`/`strict` routing |
| `OPENROUTER_PROVIDER_ORDER` | unset | Optional provider order passed through environment |
| `HF_NAMESPACE` | `tohio` | Default Hugging Face namespace |
| `HF_REPO` | unset | Explicit consolidated repo override where supported |
| `HF_PRIVATE` | unset | `true`, `yes`, or `1` for private publication |

## Qualification

```bash
QUALIFY_MODEL=<model> make model-qualify-pretrain
QUALIFY_MODEL=<model> make model-qualify-sft
QUALIFY_MODEL=<model> make model-qualify-dpo
QUALIFY_MODEL=<model> make model-qualify-distillation-sft
QUALIFY_MODEL=<model> make model-qualify-distillation-dpo
QUALIFY_MODEL=<model> make model-qualify-all
```

Advanced explicit role selection remains available:

```bash
QUALIFY_MODEL=<model> QUALIFY_ROLES=sft-generator,sft-judge,sft-reviewer make model-qualify
```

Qualification uses strict structured output like production. Optional reasoning is disabled; mandatory reasoning is unsuitable.

## Pretrain Variables

| Variable | Default |
|---|---|
| `PRETRAIN_RUN` | `pretrain-smoke-001` |
| `PRETRAIN_TARGET_RUN` | `pretrain-target-001` |
| `PRETRAIN_TOKENS` | `100000` |
| `PRETRAIN_TARGET_TOKENS` | `1000000` |
| `PRETRAIN_BATCH_SIZE` | `32` |
| `PRETRAIN_CONCURRENCY` | `1` |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` |
| `PRETRAIN_MODEL` | `openai/gpt-5.6-luna-pro` |
| `PRETRAIN_JUDGE_MODEL` | `google/gemma-4-31b-it` |
| `PRETRAIN_REVIEWER_MODEL` | `openai/gpt-5.6-luna-pro` |
| `PRETRAIN_JUDGE_BATCH_SIZE` | `10` |
| `PRETRAIN_REVIEWER_BATCH_SIZE` | `10` |
| `PRETRAIN_QUALITY_CONCURRENCY` | `8` |
| `PRETRAIN_MAX_BACKFILL_ROUNDS` | `4` |

## Generic SFT Variables

| Variable | Default |
|---|---|
| `SFT_RUN` | `sft-smoke-001` |
| `SFT_GENERATION_RUN` | `sft-production-001` |
| `SFT_FAMILIES` | `all` |
| `SFT_SMOKE_FAMILIES` | `grounded_qa_and_reading` |
| `SFT_SEEDS` | `1` |
| `SFT_DERIVATIONS_PER_SEED` | `30` |
| `SFT_TASKS_PER_DERIVATION` | `15` |
| `SFT_ANSWER_BATCH_SIZE` | `4` |
| `SFT_JUDGE_BATCH_SIZE` | `10` |
| `SFT_REVIEWER_BATCH_SIZE` | `10` |
| `SFT_CONCURRENCY` | `8` |
| `SFT_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` |
| `SFT_TASK_MODEL` | `deepseek/deepseek-v4-flash` |
| `SFT_ANSWER_MODEL` | `deepseek/deepseek-v4-flash` |
| `SFT_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` |
| `SFT_REVIEWER_MODEL` | `google/gemma-4-31b-it` |
| `SFT_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` |
| `SFT_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-sft` unless overridden |

## Generic DPO Variables

| Variable | Default |
|---|---|
| `DPO_RUN` | `dpo-smoke-001` |
| `DPO_GENERATION_RUN` | `dpo-candidate-001` |
| `DPO_PREFERENCE_DIMENSIONS` | `all` |
| `DPO_SMOKE_PREFERENCE_DIMENSIONS` | `instruction_adherence` |
| `DPO_SEEDS` | `1` |
| `DPO_DERIVATIONS_PER_SEED` | `30` |
| `DPO_TASKS_PER_DERIVATION` | `15` |
| `DPO_PAIR_BATCH_SIZE` | `4` |
| `DPO_JUDGE_BATCH_SIZE` | `10` |
| `DPO_REVIEWER_BATCH_SIZE` | `10` |
| `DPO_CONCURRENCY` | `8` |
| `DPO_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` |
| `DPO_TASK_MODEL` | `deepseek/deepseek-v4-flash` |
| `DPO_PAIR_MODEL` | `deepseek/deepseek-v4-flash` |
| `DPO_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` |
| `DPO_REVIEWER_MODEL` | `google/gemma-4-31b-it` |
| `DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` |
| `DPO_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-dpo` unless overridden |

## Distillation SFT Variables

| Variable | Default |
|---|---|
| `DISTILLATION_SFT_RUN` | `distillation-sft-smoke-001` |
| `DISTILLATION_SFT_GENERATION_RUN` | `distillation-sft-production-001` |
| `DISTILLATION_SFT_SIGNALS` | `all` |
| `DISTILLATION_SFT_SMOKE_SIGNALS` | `debugging` |
| `DISTILLATION_SFT_SEEDS` | `1` |
| `DISTILLATION_SFT_DERIVATIONS_PER_SEED` | `30` |
| `DISTILLATION_SFT_TASKS_PER_DERIVATION` | `15` |
| `DISTILLATION_SFT_ANSWER_BATCH_SIZE` | `4` |
| `DISTILLATION_SFT_JUDGE_BATCH_SIZE` | `10` |
| `DISTILLATION_SFT_REVIEWER_BATCH_SIZE` | `10` |
| `DISTILLATION_SFT_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` |
| `DISTILLATION_SFT_TASK_MODEL` | `deepseek/deepseek-v4-flash` |
| `DISTILLATION_SFT_ANSWER_MODEL` | `deepseek/deepseek-v4-flash` |
| `DISTILLATION_SFT_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` |
| `DISTILLATION_SFT_REVIEWER_MODEL` | `google/gemma-4-31b-it` |

## Distillation DPO Variables

| Variable | Default |
|---|---|
| `DISTILLATION_DPO_RUN` | `distillation-dpo-smoke-001` |
| `DISTILLATION_DPO_TARGET_RUN` | `distillation-dpo-production-001` |
| `DISTILLATION_DPO_DIMENSIONS` | `all` |
| `DISTILLATION_DPO_SMOKE_DIMENSIONS` | `factual_accuracy` |
| `DISTILLATION_DPO_SEEDS` | `1` |
| `DISTILLATION_DPO_DERIVATIONS_PER_SEED` | `30` |
| `DISTILLATION_DPO_TASKS_PER_DERIVATION` | `15` |
| `DISTILLATION_DPO_PAIR_BATCH_SIZE` | `4` |
| `DISTILLATION_DPO_JUDGE_BATCH_SIZE` | `10` |
| `DISTILLATION_DPO_REVIEWER_BATCH_SIZE` | `10` |
| `DISTILLATION_DPO_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` |
| `DISTILLATION_DPO_TASK_MODEL` | `deepseek/deepseek-v4-flash` |
| `DISTILLATION_DPO_PAIR_MODEL` | `deepseek/deepseek-v4-flash` |
| `DISTILLATION_DPO_JUDGE_MODEL` | `google/gemma-4-31b-it` |
| `DISTILLATION_DPO_REVIEWER_MODEL` | `openai/gpt-5.6-luna-pro` |
| `DISTILLATION_DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` |
| `DISTILLATION_DPO_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-distillation-dpo` |

## Maintenance

```bash
make test
make clean
```

There are no supported legacy generation, preflight, manual adjudication, cost-estimation, or Hugging Face deletion Make targets.
