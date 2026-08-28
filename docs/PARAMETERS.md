# Parameter Reference

Exhaustive reference for the user-facing Make variables supported by the current five-dataset production workflows.

The Makefile is a thin operational wrapper. It does **not** remove configurability: Make variables map directly to the arguments passed into the five production pipelines and their report/push commands.

## How to Override a Parameter

Use Make variables on the command line:

```bash
SFT_FAMILIES=programming \
SFT_CONCURRENCY=8 \
SFT_TASK_MODEL=deepseek/deepseek-v4-flash \
make sft-generate
```

Variables supplied this way override the Makefile defaults for that invocation.

## Important Distinctions

- **batch size != concurrency** — batch size is the number of logical items requested in one structured model call; concurrency is the number of calls allowed in flight at once.
- **seeds != random seed** — `*_SEEDS` is the number of semantic starting points generated per selected family/signal/dimension.
- **task model != answer/pair model** — the task model writes the user-facing task; the answer/pair model writes the training response(s).
- **judge != reviewer** — judge is the first semantic quality gate; reviewer independently re-checks judge-accepted rows.
- **planned candidates != final accepted rows** — novelty, deterministic validation, judge, reviewer, and final dedup can reduce realized output.
- **pretraining target tokens != raw generated tokens** — pretraining completes on post-review, post-dedup accepted tokens.
- **family/signal/dimension != public dataset** — these are internal coverage partitions inside one public product.
- **routing mode != provider order** — routing mode defines fallback behavior; provider order defines preferred provider ordering.

## Shared Runtime and Provider Parameters

| Parameter | Default | Meaning | When to change it |
|---|---|---|---|
| `PYTHON` | `python` | Python executable used by all Make targets. | Override when the desired interpreter is not on `python`, e.g. `PYTHON=.venv/bin/python`. |
| `MODEL` | `deepseek/deepseek-v4-flash` | Generic fallback model used by qualification defaults. Dataset pipelines have their own role-specific model variables. | Primarily when using `model-qualify` without setting `QUALIFY_MODEL`. |
| `MAX_TOKENS` | `4096` | Shared fallback output-token ceiling. Currently used as the default for `PRETRAIN_MAX_TOKENS`. | Increase only when the selected role genuinely needs larger structured outputs and the provider supports it. |
| `OPENROUTER_ROUTING_MODE` | `auto` | Provider routing policy: `auto`, `prefer`, or `strict`. | Use `prefer` to favor a provider while keeping fallback; `strict` to pin one provider. |
| `OPENROUTER_PROVIDER` | unset | Named provider used with `prefer` or `strict` routing. | Set when a specific provider is preferred/required. |
| `OPENROUTER_PROVIDER_ORDER` | unset | Ordered provider preference list passed through the environment to OpenRouter routing. | Set when you want fallback but need a stable provider priority order. |
| `OPENROUTER_PROVIDER_ONLY` | unset | Optional provider allow-list passed through to routing. | Use to restrict eligible providers without hard-coding provider logic in code. |
| `OPENROUTER_PROVIDER_IGNORE` | unset | Optional provider deny-list passed through to routing. | Use when one or more providers are known to be unsuitable or failing. |
| `OPENROUTER_PROVIDER_SORT` | unset | Optional provider sorting preference passed through to routing. | Use only when you need explicit OpenRouter provider sorting behavior. |

Example:

```bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
make distillation-sft-generate
```

## Model Qualification Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `QUALIFY_MODEL` | `$(MODEL)` | Model id to qualify. |
| `QUALIFY_ROLES` | `all` | Qualification role group. Use `all`, `pretrain`, `sft`, `dpo`, `distillation-sft`, `distillation-dpo`, or explicit role names supported by `qualify_model.py`. |
| `QUALIFY_MAX_TOKENS` | `512` | Maximum output tokens for each qualification probe. |
| `QUALIFY_OUTPUT` | `data/model-qualification/<model>.json` | JSON report path for qualification results. |

Qualification verifies the strict structured-output request path and the repository's reasoning-disable policy. A reasoning-capable model is eligible only when reasoning can be disabled successfully for a live structured request.

---

# Pretraining Parameters

Pretraining is sized by **accepted tokens**, not by a fixed row count.

## Run, Config, and Output

| Parameter | Default | Meaning |
|---|---|---|
| `CONFIG_FILE` | `configs/synthetic.yaml` | Generated pretraining config consumed by the pipeline/report commands. |
| `DATA_DIR` | `data/runs` | Root for pretraining run artifacts used by inspect/clean. |
| `PROFILE` | `balanced` | Pretraining configuration profile passed to `configs/configure_synthetic.py`. |
| `PRETRAIN_RUN` | `pretrain-smoke-001` | Run id used by `pretrain-smoke`. |
| `PRETRAIN_TARGET_RUN` | `pretrain-target-001` | Run id used by `pretrain-generate`. |
| `PRETRAIN_REPORT_RUN` | `$(PRETRAIN_RUN)` | Run id consumed by `pretrain-report` and, indirectly, by the default inspect/push workflow. |
| `PRETRAIN_INSPECT_RUN` | `$(PRETRAIN_REPORT_RUN)` | Run id displayed by `pretrain-inspect`. |
| `PRETRAIN_SIGNAL` | unset | Optional single pretraining signal filter. Empty means all configured signals. |

Supported pretraining signals are `arithmetic`, `task_code`, `educational_qa_mcq_math`, `educational_qa_mcq_general`, and `factual_restraint`.

## Sizing and Throughput

| Parameter | Default | Meaning | Effect of increasing it |
|---|---:|---|---|
| `PRETRAIN_TOKENS` | `100000` | Final accepted-token target for `pretrain-smoke`. | Larger smoke/output volume and more provider cost. |
| `PRETRAIN_TARGET_TOKENS` | `1000000` | Final accepted-token target for `pretrain-generate`. | More final data; pipeline generates/backfills enough candidates to meet the post-quality target. |
| `PRETRAIN_BATCH_SIZE` | `32` | Grounded generation batch size written into the synthetic config. | Fewer, larger generator requests; may raise failure/throttling risk. |
| `PRETRAIN_CONCURRENCY` | `1` | Grounded generation concurrency for smoke. | Faster smoke until provider limits are reached. |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | Grounded generation concurrency for production. | Faster generation but greater provider pressure. |
| `PRETRAIN_QUALITY_CONCURRENCY` | `8` | Concurrent semantic judge/reviewer requests. | Faster quality stages but more rate-limit pressure. |
| `PRETRAIN_STAGE_BATCH_ATTEMPTS` | `3` | Attempts for a failed semantic-quality batch before runtime isolation/splitting. | More retry tolerance before splitting. |
| `PRETRAIN_MAX_BACKFILL_ROUNDS` | `4` | Maximum post-quality rounds used to fill an accepted-token deficit. | More opportunity to hit target at additional cost/time. |

The pretraining target is evaluated after:

```text
generation -> deterministic validation -> judge -> reviewer -> final dedup -> accepted-token count
```

## Model Roles

| Parameter | Default | Role |
|---|---|---|
| `PRETRAIN_MODEL` | `openai/gpt-5.6-luna-pro` | Renders deterministic grounded artifacts into final language-bearing pretraining records. |
| `PRETRAIN_JUDGE_MODEL` | `google/gemma-4-31b-it` | First semantic quality gate. |
| `PRETRAIN_REVIEWER_MODEL` | `openai/gpt-5.6-luna-pro` | Independent review of judge-accepted rows. |

## Token Limits and Quality Batch Sizes

| Parameter | Default | Meaning |
|---|---:|---|
| `PRETRAIN_MAX_TOKENS` | `$(MAX_TOKENS)` | Generator output-token ceiling. |
| `PRETRAIN_JUDGE_MAX_TOKENS` | `4096` | Judge response ceiling. |
| `PRETRAIN_REVIEWER_MAX_TOKENS` | `4096` | Reviewer response ceiling. |
| `PRETRAIN_JUDGE_BATCH_SIZE` | `10` | Candidate rows adjudicated per judge request. |
| `PRETRAIN_REVIEWER_BATCH_SIZE` | `10` | Judge-accepted rows reviewed per reviewer request. |

## Reporting / Diversity

| Parameter | Default | Meaning |
|---|---:|---|
| `PRETRAIN_DIVERSITY_SAMPLE_SIZE` | `10000` | Maximum final rows sampled by the diversity report. |
| `PRETRAIN_DIVERSITY_THRESHOLD` | `0.80` | Near-duplicate similarity threshold used by the final diversity audit. |

---

# Generic SFT Parameters

## Sizing Model

For each selected SFT family, planned task candidates are approximately:

```text
SFT_SEEDS x SFT_DERIVATIONS_PER_SEED x SFT_TASKS_PER_DERIVATION
```

With defaults:

```text
1 x 30 x 15 = 450 planned task candidates per selected family
```

This is **not** a guaranteed final row count. Task novelty, structured-output failures, deterministic validation, judge/reviewer rejection, and final dedup can reduce the accepted count.

## Coverage, Run, and Output

| Parameter | Default | Meaning |
|---|---|---|
| `SFT_RUN` | `sft-smoke-001` | Run id for `sft-smoke`. |
| `SFT_GENERATION_RUN` | `sft-production-001` | Run id for `sft-generate`. |
| `SFT_REPORT_RUN` | `$(SFT_RUN)` | Run id used by `sft-report`. |
| `SFT_INSPECT_RUN` | `$(SFT_REPORT_RUN)` | Run id displayed by `sft-inspect`. |
| `SFT_RUN_ROOT` | `data/sft/runs` | Root directory for SFT runs. |
| `SFT_FAMILIES` | `all` | Families selected for production generation. Use `all` or one/more family names accepted by the pipeline. |
| `SFT_SMOKE_FAMILIES` | `grounded_qa_and_reading` | Default family used by the smoke target when a production family override is not supplied. |
| `SFT_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Evaluation holdout registry used by reporting/publication checks. |

See [Generation Families](GENERATION_FAMILIES.md#generic-sft-families) for valid family names and intended coverage.

## Sizing and Expansion

| Parameter | Default | Meaning | Effect of increasing it |
|---|---:|---|---|
| `SFT_SEEDS` | `1` | Number of independent semantic starting points per selected family. | More conceptual breadth and more generation calls. |
| `SFT_DERIVATIONS_PER_SEED` | `30` | Semantic derivations generated from each seed in production. | More semantic expansion and planned task volume. |
| `SFT_TASKS_PER_DERIVATION` | `15` | Concrete tasks generated from each accepted derivation in production. | More planned task candidates per derivation. |
| `SFT_SMOKE_DERIVATIONS_PER_SEED` | `1` | Smoke-only derivations per seed. | Larger smoke coverage. |
| `SFT_SMOKE_TASKS_PER_DERIVATION` | `2` | Smoke-only tasks per derivation. | Larger smoke coverage. |

`SFT_SEEDS` is a **count**, not a PRNG seed.

## Model Roles

| Parameter | Default | Role |
|---|---|---|
| `SFT_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` | Expands a family specification into varied semantic derivations. |
| `SFT_TASK_MODEL` | `deepseek/deepseek-v4-flash` | Converts a derivation into concrete user-facing tasks. It does not write the assistant answer. |
| `SFT_ANSWER_MODEL` | `deepseek/deepseek-v4-flash` | Generates the assistant response for an accepted task. |
| `SFT_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` | First semantic quality decision after deterministic validation. |
| `SFT_REVIEWER_MODEL` | `google/gemma-4-31b-it` | Independently reviews judge-accepted rows before final acceptance. |

## Token Ceilings

| Parameter | Default | Meaning |
|---|---:|---|
| `SFT_DERIVATION_MAX_TOKENS` | `4096` | Output ceiling for derivation calls. |
| `SFT_TASK_MAX_TOKENS` | `4096` | Output ceiling for task-generation calls. |
| `SFT_ANSWER_MAX_TOKENS` | `4096` | Output ceiling for assistant-answer calls. |
| `SFT_JUDGE_MAX_TOKENS` | `4096` | Output ceiling for judge calls. |
| `SFT_REVIEWER_MAX_TOKENS` | `512` | Output ceiling for reviewer calls. |

These are request ceilings, not target training-example lengths.

## Batching, Concurrency, and Failure Recovery

| Parameter | Default | Meaning |
|---|---:|---|
| `SFT_ANSWER_BATCH_SIZE` | `4` | Number of SFT answer items requested per answer-model call. |
| `SFT_JUDGE_BATCH_SIZE` | `10` | Number of candidates adjudicated per judge call. |
| `SFT_REVIEWER_BATCH_SIZE` | `10` | Number of judge-accepted candidates reviewed per reviewer call. |
| `SFT_CONCURRENCY` | `8` | Maximum model-stage requests in flight concurrently. |
| `SFT_CARDINALITY_FILL_ATTEMPTS` | `3` | Additional attempts to fill missing structured items when a call returns fewer valid items than requested. |
| `SFT_STAGE_BATCH_ATTEMPTS` | `3` | Attempts for a failed batch before the runtime recursively splits/isolate failures. |

## Novelty

| Parameter | Default | Meaning |
|---|---:|---|
| `SFT_JACCARD_THRESHOLD` | `0.82` | Near-duplicate task threshold based on normalized shingle overlap. |
| `SFT_SEQUENCE_THRESHOLD` | `0.90` | Near-duplicate task threshold based on normalized sequence similarity. |

Higher thresholds allow closer candidates before rejection; lower thresholds make novelty filtering stricter.

## Publication

| Parameter | Default | Meaning |
|---|---|---|
| `SFT_PUSH_RUN` | `$(SFT_REPORT_RUN)` | Run uploaded by `sft-push`. |
| `SFT_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-sft` unless `HF_REPO` is set | Destination Hugging Face dataset repo. |

---

# Generic DPO Parameters

## Sizing Model

For each selected preference dimension:

```text
DPO_SEEDS x DPO_DERIVATIONS_PER_SEED x DPO_TASKS_PER_DERIVATION
```

Default planned volume:

```text
1 x 30 x 15 = 450 planned preference tasks per selected dimension
```

The final accepted pair count can be smaller after novelty, pair validation, judge/reviewer rejection, holdout checks, and exact triple dedup.

## Coverage, Run, and Output

| Parameter | Default | Meaning |
|---|---|---|
| `DPO_RUN` | `dpo-smoke-001` | Run id for `dpo-smoke`. |
| `DPO_GENERATION_RUN` | `dpo-candidate-001` | Run id for `dpo-generate`. |
| `DPO_REPORT_RUN` | `$(DPO_RUN)` | Run id used by `dpo-report`. |
| `DPO_INSPECT_RUN` | `$(DPO_REPORT_RUN)` | Run id displayed by `dpo-inspect`. |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Root directory for DPO runs. |
| `DPO_PREFERENCE_DIMENSIONS` | `all` | Preference dimensions selected for production. |
| `DPO_SMOKE_PREFERENCE_DIMENSIONS` | `instruction_adherence` | Default smoke dimension. |
| `DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Evaluation holdout registry. |

See [Generation Families](GENERATION_FAMILIES.md#generic-dpo-preference-dimensions) for valid dimension names.

## Sizing and Expansion

| Parameter | Default | Meaning |
|---|---:|---|
| `DPO_SEEDS` | `1` | Independent semantic starting points per selected dimension. |
| `DPO_DERIVATIONS_PER_SEED` | `30` | Production derivations per seed. |
| `DPO_TASKS_PER_DERIVATION` | `15` | Production tasks per accepted derivation. |
| `DPO_SMOKE_DERIVATIONS_PER_SEED` | `2` | Smoke derivations per seed. |
| `DPO_SMOKE_TASKS_PER_DERIVATION` | `2` | Smoke tasks per derivation. |

## Model Roles

| Parameter | Default | Role |
|---|---|---|
| `DPO_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` | Produces semantic derivations for the selected preference dimension. |
| `DPO_TASK_MODEL` | `deepseek/deepseek-v4-flash` | Produces concrete user tasks. |
| `DPO_PAIR_MODEL` | `deepseek/deepseek-v4-flash` | Produces both chosen and deliberately weaker rejected branches. |
| `DPO_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` | First semantic preference-quality gate. |
| `DPO_REVIEWER_MODEL` | `google/gemma-4-31b-it` | Independent review of judge-accepted pairs. |

## Token Ceilings

| Parameter | Default |
|---|---:|
| `DPO_DERIVATION_MAX_TOKENS` | `4096` |
| `DPO_TASK_MAX_TOKENS` | `4096` |
| `DPO_PAIR_MAX_TOKENS` | `4096` |
| `DPO_JUDGE_MAX_TOKENS` | `4096` |
| `DPO_REVIEWER_MAX_TOKENS` | `512` |

Each is the maximum output-token budget for the named stage.

## Batching, Concurrency, and Failure Recovery

| Parameter | Default | Meaning |
|---|---:|---|
| `DPO_PAIR_BATCH_SIZE` | `4` | Preference tasks requested per pair-generation call. |
| `DPO_JUDGE_BATCH_SIZE` | `10` | Candidate pairs adjudicated per judge call. |
| `DPO_REVIEWER_BATCH_SIZE` | `10` | Judge-accepted pairs reviewed per reviewer call. |
| `DPO_CONCURRENCY` | `8` | Maximum concurrent model-stage calls. |
| `DPO_CARDINALITY_FILL_ATTEMPTS` | `3` | Attempts to fill missing structured items. |
| `DPO_STAGE_BATCH_ATTEMPTS` | `3` | Attempts before failed batches are recursively split/isolate. |

## Novelty

| Parameter | Default | Meaning |
|---|---:|---|
| `DPO_JACCARD_THRESHOLD` | `0.82` | Task near-duplicate Jaccard threshold. |
| `DPO_SEQUENCE_THRESHOLD` | `0.90` | Task near-duplicate sequence threshold. |

## Publication

| Parameter | Default | Meaning |
|---|---|---|
| `DPO_PUSH_RUN` | `$(DPO_REPORT_RUN)` | Run uploaded by `dpo-push`. |
| `DPO_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-dpo` unless `HF_REPO` is set | Destination Hugging Face dataset repo. |

---

# Distillation SFT Parameters

## Sizing Model

For each selected distillation signal:

```text
DISTILLATION_SFT_SEEDS
x DISTILLATION_SFT_DERIVATIONS_PER_SEED
x DISTILLATION_SFT_TASKS_PER_DERIVATION
```

Default planned production volume:

```text
1 x 30 x 15 = 450 planned teacher-response tasks per selected signal
```

Final accepted rows can be smaller because this pipeline also applies response-level novelty and final prompt/response uniqueness.

## Coverage, Run, and Output

| Parameter | Default | Meaning |
|---|---|---|
| `DISTILLATION_SFT_RUN` | `distillation-sft-smoke-001` | Run id for smoke. |
| `DISTILLATION_SFT_GENERATION_RUN` | `distillation-sft-production-001` | Run id for production. |
| `DISTILLATION_SFT_REPORT_RUN` | `$(DISTILLATION_SFT_RUN)` | Run id used by the report target. |
| `DISTILLATION_SFT_INSPECT_RUN` | `$(DISTILLATION_SFT_REPORT_RUN)` | Run id displayed by inspect. |
| `DISTILLATION_SFT_RUN_ROOT` | `data/distillation/runs` | Root directory for Distillation-SFT runs. |
| `DISTILLATION_SFT_SIGNALS` | `all` | Signals selected for production. |
| `DISTILLATION_SFT_SMOKE_SIGNALS` | `debugging` | Default smoke signal. |
| `DISTILLATION_SFT_DATASET_NAME` | `SLM Synthetic Distillation` | Display name used when building the run dataset card. |

See [Generation Families](GENERATION_FAMILIES.md#distillation-sft-signals) for valid signals.

## Sizing and Expansion

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_SFT_SEEDS` | `1` | Independent semantic starting points per selected signal. |
| `DISTILLATION_SFT_DERIVATIONS_PER_SEED` | `30` | Production semantic derivations per seed. |
| `DISTILLATION_SFT_TASKS_PER_DERIVATION` | `15` | Student-facing prompts per accepted derivation. |
| `DISTILLATION_SFT_SMOKE_DERIVATIONS_PER_SEED` | `1` | Smoke derivations per seed. |
| `DISTILLATION_SFT_SMOKE_TASKS_PER_DERIVATION` | `2` | Smoke prompts per derivation. |

## Model Roles

| Parameter | Default | Role |
|---|---|---|
| `DISTILLATION_SFT_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` | Produces diverse semantic derivations for the selected signal. |
| `DISTILLATION_SFT_TASK_MODEL` | `deepseek/deepseek-v4-flash` | Converts a derivation into the **student-facing prompt**. |
| `DISTILLATION_SFT_ANSWER_MODEL` | `deepseek/deepseek-v4-flash` | Teacher model that writes the response the downstream student will learn from. |
| `DISTILLATION_SFT_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` | First semantic quality gate over validated teacher responses. |
| `DISTILLATION_SFT_REVIEWER_MODEL` | `google/gemma-4-31b-it` | Independent review of judge-accepted teacher responses. |

## Token Ceilings

| Parameter | Default |
|---|---:|
| `DISTILLATION_SFT_DERIVATION_MAX_TOKENS` | `4096` |
| `DISTILLATION_SFT_TASK_MAX_TOKENS` | `4096` |
| `DISTILLATION_SFT_ANSWER_MAX_TOKENS` | `4096` |
| `DISTILLATION_SFT_JUDGE_MAX_TOKENS` | `4096` |
| `DISTILLATION_SFT_REVIEWER_MAX_TOKENS` | `512` |

Each is the output-token ceiling for its named stage.

## Batching, Concurrency, and Failure Recovery

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_SFT_ANSWER_BATCH_SIZE` | `4` | Student prompts answered per teacher-response call. |
| `DISTILLATION_SFT_JUDGE_BATCH_SIZE` | `10` | Validated teacher responses adjudicated per judge call. |
| `DISTILLATION_SFT_REVIEWER_BATCH_SIZE` | `10` | Judge-accepted responses reviewed per reviewer call. |
| `DISTILLATION_SFT_CONCURRENCY` | `8` | Maximum concurrent model-stage calls. |
| `DISTILLATION_SFT_CARDINALITY_FILL_ATTEMPTS` | `3` | Attempts to fill missing structured items. |
| `DISTILLATION_SFT_STAGE_BATCH_ATTEMPTS` | `3` | Attempts before failed model batches are recursively split/isolate. |

## Novelty

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_SFT_JACCARD_THRESHOLD` | `0.82` | Student-prompt near-duplicate Jaccard threshold. |
| `DISTILLATION_SFT_SEQUENCE_THRESHOLD` | `0.90` | Student-prompt near-duplicate sequence threshold. |

Response-level novelty is an additional Distillation-SFT quality rule implemented by the pipeline and is not replaced by these task-novelty thresholds.

## Publication

| Parameter | Default | Meaning |
|---|---|---|
| `DISTILLATION_SFT_PUSH_RUN` | `$(DISTILLATION_SFT_REPORT_RUN)` | Run uploaded by `distillation-sft-push`. |
| `DISTILLATION_SFT_HF_REPO` | `HF_REPO` when set, otherwise `$(HF_NAMESPACE)/slm-synthetic-distillation-sft` | Destination Hugging Face repo. |

---

# Distillation DPO Parameters

## Sizing Model

For each selected preference dimension:

```text
DISTILLATION_DPO_SEEDS
x DISTILLATION_DPO_DERIVATIONS_PER_SEED
x DISTILLATION_DPO_TASKS_PER_DERIVATION
```

Defaults:

```text
1 x 30 x 15 = 450 planned preference tasks per selected dimension
```

Final accepted rows can be smaller after task novelty, pair validation, five-gate judge, reviewer, holdout checks, and final triple dedup.

## Coverage, Run, and Output

| Parameter | Default | Meaning |
|---|---|---|
| `DISTILLATION_DPO_RUN` | `distillation-dpo-smoke-001` | Smoke run id. |
| `DISTILLATION_DPO_TARGET_RUN` | `distillation-dpo-production-001` | Production run id. |
| `DISTILLATION_DPO_REPORT_RUN` | `$(DISTILLATION_DPO_RUN)` | Run id used by report. |
| `DISTILLATION_DPO_INSPECT_RUN` | `$(DISTILLATION_DPO_REPORT_RUN)` | Run id displayed by inspect. |
| `DISTILLATION_DPO_RUN_ROOT` | `data/distillation-dpo/runs` | Root directory for Distillation-DPO runs. |
| `DISTILLATION_DPO_DIMENSIONS` | `all` | Preference dimensions selected for production. |
| `DISTILLATION_DPO_SMOKE_DIMENSIONS` | `factual_accuracy` | Default smoke dimension. |
| `DISTILLATION_DPO_DATASET_NAME` | `SLM Synthetic Distillation DPO` | Display name used in the run dataset card. |
| `DISTILLATION_DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Evaluation holdout registry. |

See [Generation Families](GENERATION_FAMILIES.md#distillation-dpo-dimensions) for valid dimensions.

## Sizing and Expansion

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_DPO_SEEDS` | `1` | Independent semantic starting points per selected dimension. |
| `DISTILLATION_DPO_DERIVATIONS_PER_SEED` | `30` | Production derivations per seed. |
| `DISTILLATION_DPO_TASKS_PER_DERIVATION` | `15` | Preference tasks per accepted derivation. |
| `DISTILLATION_DPO_SMOKE_DERIVATIONS_PER_SEED` | `1` | Smoke derivations per seed. |
| `DISTILLATION_DPO_SMOKE_TASKS_PER_DERIVATION` | `2` | Smoke tasks per derivation. |

## Model Roles

| Parameter | Default | Role |
|---|---|---|
| `DISTILLATION_DPO_DERIVATION_MODEL` | `openai/gpt-5.6-luna-pro` | Produces semantic derivations using Distillation-DPO-specific guidance. |
| `DISTILLATION_DPO_TASK_MODEL` | `deepseek/deepseek-v4-flash` | Produces concrete preference tasks. |
| `DISTILLATION_DPO_PAIR_MODEL` | `deepseek/deepseek-v4-flash` | Produces chosen and controlled-defect rejected branches. |
| `DISTILLATION_DPO_JUDGE_MODEL` | `google/gemma-4-31b-it` | Runs the Distillation-DPO five-gate judge contract. |
| `DISTILLATION_DPO_REVIEWER_MODEL` | `openai/gpt-5.6-luna-pro` | Independently reviews judge-accepted pairs. |

The judge role is intentionally different from generic DPO; do not assume the two pipelines are semantically interchangeable.

## Token Ceilings

| Parameter | Default |
|---|---:|
| `DISTILLATION_DPO_DERIVATION_MAX_TOKENS` | `4096` |
| `DISTILLATION_DPO_TASK_MAX_TOKENS` | `4096` |
| `DISTILLATION_DPO_PAIR_MAX_TOKENS` | `4096` |
| `DISTILLATION_DPO_JUDGE_MAX_TOKENS` | `4096` |
| `DISTILLATION_DPO_REVIEWER_MAX_TOKENS` | `512` |

## Batching, Concurrency, and Failure Recovery

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_DPO_PAIR_BATCH_SIZE` | `4` | Preference tasks handled per pair-generation call. |
| `DISTILLATION_DPO_JUDGE_BATCH_SIZE` | `10` | Candidate pairs handled per five-gate judge call. |
| `DISTILLATION_DPO_REVIEWER_BATCH_SIZE` | `10` | Judge-accepted pairs handled per reviewer call. |
| `DISTILLATION_DPO_CONCURRENCY` | `8` | Maximum concurrent model-stage calls. |
| `DISTILLATION_DPO_CARDINALITY_FILL_ATTEMPTS` | `3` | Attempts to fill missing structured items. |
| `DISTILLATION_DPO_STAGE_BATCH_ATTEMPTS` | `3` | Attempts before failed batches are recursively split/isolate. |

## Novelty

| Parameter | Default | Meaning |
|---|---:|---|
| `DISTILLATION_DPO_JACCARD_THRESHOLD` | `0.82` | Task near-duplicate Jaccard threshold. |
| `DISTILLATION_DPO_SEQUENCE_THRESHOLD` | `0.90` | Task near-duplicate sequence threshold. |

## Publication

| Parameter | Default | Meaning |
|---|---|---|
| `DISTILLATION_DPO_PUSH_RUN` | `$(DISTILLATION_DPO_REPORT_RUN)` | Run uploaded by push. |
| `DISTILLATION_DPO_HF_REPO` | `$(HF_NAMESPACE)/slm-synthetic-distillation-dpo` | Destination Hugging Face repo. |

---

# Shared Hugging Face Publication Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `HF_REPO` | unset | Explicit repository override. Used directly by pretraining and as an override by SFT/Distillation-SFT; generic DPO also honors it through its computed default. |
| `HF_NAMESPACE` | `tohio` | Namespace used when a dataset-specific repo is not explicitly supplied. |
| `HF_PRIVATE` | unset/false | When `true`, `yes`, or `1`, push commands create/update the dataset as private where supported by the package push implementation. |

`HF_TOKEN` is an environment credential, not a Make variable. It is required only when publishing.

# Fully Explicit Examples

## Generic SFT

```bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
SFT_GENERATION_RUN=sft-production-001 \
SFT_FAMILIES=all \
SFT_SEEDS=1 \
SFT_DERIVATIONS_PER_SEED=30 \
SFT_TASKS_PER_DERIVATION=15 \
SFT_CONCURRENCY=8 \
SFT_ANSWER_BATCH_SIZE=4 \
SFT_JUDGE_BATCH_SIZE=10 \
SFT_REVIEWER_BATCH_SIZE=10 \
SFT_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
SFT_TASK_MODEL=deepseek/deepseek-v4-flash \
SFT_ANSWER_MODEL=deepseek/deepseek-v4-flash \
SFT_JUDGE_MODEL=nvidia/nemotron-3.5-lightning \
SFT_REVIEWER_MODEL=google/gemma-4-31b-it \
make sft-generate
```

## Distillation SFT

```bash
OPENROUTER_ROUTING_MODE=auto \
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
DISTILLATION_SFT_SIGNALS=cloud \
DISTILLATION_SFT_SEEDS=1 \
DISTILLATION_SFT_DERIVATIONS_PER_SEED=30 \
DISTILLATION_SFT_TASKS_PER_DERIVATION=15 \
DISTILLATION_SFT_CONCURRENCY=8 \
DISTILLATION_SFT_ANSWER_BATCH_SIZE=4 \
DISTILLATION_SFT_JUDGE_BATCH_SIZE=10 \
DISTILLATION_SFT_REVIEWER_BATCH_SIZE=10 \
DISTILLATION_SFT_DERIVATION_MODEL=openai/gpt-5.6-luna-pro \
DISTILLATION_SFT_TASK_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_SFT_ANSWER_MODEL=deepseek/deepseek-v4-flash \
DISTILLATION_SFT_JUDGE_MODEL=nvidia/nemotron-3.5-lightning \
DISTILLATION_SFT_REVIEWER_MODEL=google/gemma-4-31b-it \
make distillation-sft-generate
```

See [Command Reference](COMMANDS.md) for target-oriented examples and [Generation Workflow](GENERATION_WORKFLOW.md) for stage behavior and run artifacts.
