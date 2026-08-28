# Command Reference

Supported Make targets and operational examples. For the exhaustive meaning of every Make variable—including sizing, model roles, batching, concurrency, retries, novelty thresholds, output roots, and publication controls—see [Parameter Reference](PARAMETERS.md).

The Make targets are wrappers around the production pipeline arguments. A short command such as `make sft-generate` uses Makefile defaults; supplying `SFT_*`, `DPO_*`, `DISTILLATION_*`, `PRETRAIN_*`, or shared OpenRouter variables overrides those defaults without changing the supported production path.

## Supported Target Matrix

Every dataset has the same five user-facing operations:

| Dataset | Smoke | Generate | Inspect | Report | Push |
|---|---|---|---|---|---|
| Pretraining | `pretrain-smoke` | `pretrain-generate` | `pretrain-inspect` | `pretrain-report` | `pretrain-push` |
| Generic SFT | `sft-smoke` | `sft-generate` | `sft-inspect` | `sft-report` | `sft-push` |
| Generic DPO | `dpo-smoke` | `dpo-generate` | `dpo-inspect` | `dpo-report` | `dpo-push` |
| Distillation SFT | `distillation-sft-smoke` | `distillation-sft-generate` | `distillation-sft-inspect` | `distillation-sft-report` | `distillation-sft-push` |
| Distillation DPO | `distillation-dpo-smoke` | `distillation-dpo-generate` | `distillation-dpo-inspect` | `distillation-dpo-report` | `distillation-dpo-push` |

Additional supported targets:

~~~text
model-qualify
model-qualify-pretrain
model-qualify-sft
model-qualify-dpo
model-qualify-distillation-sft
model-qualify-distillation-dpo
model-qualify-all
test
clean
help
~~~

## Environment

Copy `.env.sample` to `.env`.

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | live generation/qualification | OpenRouter authentication |
| `HF_TOKEN` | push only | Hugging Face dataset publication |
| `DATA_DIR` | optional | pretraining output root used by `configs/synthetic.yaml`; default behavior uses `data/runs` |

## OpenRouter Routing

The Makefile passes routing controls to all production pipelines.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_ROUTING_MODE` | `auto` | `auto`, `prefer`, or `strict` |
| `OPENROUTER_PROVIDER` | unset | provider used by `prefer` or `strict` |
| `OPENROUTER_PROVIDER_ORDER` | unset | preferred provider order exposed through environment |
| `OPENROUTER_PROVIDER_ONLY` | unset | optional provider restriction |
| `OPENROUTER_PROVIDER_IGNORE` | unset | optional provider exclusion |
| `OPENROUTER_PROVIDER_SORT` | unset | optional provider sorting behavior |

Examples:

~~~bash
# Allow OpenRouter to route normally.
make sft-smoke

# Prefer one provider but preserve fallback.
OPENROUTER_ROUTING_MODE=prefer \
OPENROUTER_PROVIDER=DeepInfra \
make sft-smoke

# Pin one provider.
OPENROUTER_ROUTING_MODE=strict \
OPENROUTER_PROVIDER=DeepInfra \
make sft-smoke

# Preserve an ordered fallback list.
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" \
make distillation-sft-smoke
~~~

## Model Qualification

Qualify a candidate model against the production structured-output contract before using it in a new role.

~~~bash
QUALIFY_MODEL=<model-id> make model-qualify-pretrain
QUALIFY_MODEL=<model-id> make model-qualify-sft
QUALIFY_MODEL=<model-id> make model-qualify-dpo
QUALIFY_MODEL=<model-id> make model-qualify-distillation-sft
QUALIFY_MODEL=<model-id> make model-qualify-distillation-dpo
~~~

Qualify all fifteen dataset roles:

~~~bash
QUALIFY_MODEL=<model-id> make model-qualify-all
~~~

Explicit role selection is also supported:

~~~bash
QUALIFY_MODEL=<model-id> \
QUALIFY_ROLES=sft-generator,sft-judge,sft-reviewer \
make model-qualify
~~~

Qualification output defaults to:

~~~text
data/model-qualification/<model-id-with-slashes-replaced>.json
~~~

Reasoning policy is fail-closed: when a model supports optional reasoning, requests disable reasoning; mandatory/non-disableable reasoning models are rejected as unsuitable. Role model ids shown below are defaults only. Any qualified model may replace any role model.

---

## How Sizing Variables Work

SFT, DPO, Distillation SFT, and Distillation DPO use a three-level expansion plan:

```text
planned tasks per selected family/signal/dimension
    = seeds x derivations per seed x tasks per derivation
```

For the production defaults, `1 x 30 x 15 = 450` planned tasks per selected coverage unit. This is a candidate plan, not a guaranteed accepted row count. Quality gates and deduplication reduce realized output.

Pretraining is different: `PRETRAIN_TARGET_TOKENS` is the final accepted-token target after deterministic validation, judge, reviewer, and final dedup.

See [Parameter Reference](PARAMETERS.md) for the exact meaning of each sizing and runtime variable.

---

## Pretraining

### Smoke

~~~bash
make pretrain-smoke
~~~

Default smoke settings:

| Variable | Default |
|---|---|
| `PRETRAIN_RUN` | `pretrain-smoke-001` |
| `PRETRAIN_TOKENS` | `100000` |
| `PRETRAIN_BATCH_SIZE` | `32` |
| `PRETRAIN_CONCURRENCY` | `1` |
| `PRETRAIN_MODEL` | `openai/gpt-5.6-luna-pro` |
| `PRETRAIN_JUDGE_MODEL` | `google/gemma-4-31b-it` |
| `PRETRAIN_REVIEWER_MODEL` | `openai/gpt-5.6-luna-pro` |

### Production

~~~bash
PRETRAIN_TARGET_RUN=pretrain-production-001 \
PRETRAIN_TARGET_TOKENS=1000000 \
make pretrain-generate
~~~

Important variables:

| Variable | Default | Purpose |
|---|---:|---|
| `PRETRAIN_TARGET_TOKENS` | `1000000` | accepted post-review token target |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | grounded generation concurrency |
| `PRETRAIN_MAX_TOKENS` | `4096` | generator request output limit |
| `PRETRAIN_TEMPERATURE` | unset | optional generator sampling temperature; omitted from provider request when unset |
| `PRETRAIN_TOP_P` | unset | optional generator nucleus-sampling value; omitted from provider request when unset |
| `PRETRAIN_JUDGE_BATCH_SIZE` | `10` | semantic judge batch size |
| `PRETRAIN_REVIEWER_BATCH_SIZE` | `10` | reviewer batch size |
| `PRETRAIN_QUALITY_CONCURRENCY` | `8` | judge/reviewer stage concurrency |
| `PRETRAIN_STAGE_BATCH_ATTEMPTS` | `3` | model-stage attempts before recursive isolation |
| `PRETRAIN_MAX_BACKFILL_ROUNDS` | `4` | maximum semantic-quality backfill rounds |
| `PRETRAIN_SIGNAL` | unset | optional single-signal run/debug filter |
| `PRETRAIN_DIVERSITY_SAMPLE_SIZE` | `10000` | bounded diversity-report sample |
| `PRETRAIN_DIVERSITY_THRESHOLD` | `0.80` | report near-duplicate threshold |

`pretrain-smoke` and `pretrain-generate` rewrite `configs/synthetic.yaml` from the template before running.

### Inspect / Report / Push

~~~bash
PRETRAIN_INSPECT_RUN=pretrain-production-001 make pretrain-inspect
PRETRAIN_REPORT_RUN=pretrain-production-001 make pretrain-report
~~~

`pretrain-push` publishes the run referenced by the current `configs/synthetic.yaml` output directory. The normal workflow is therefore to push immediately after the corresponding `pretrain-smoke` or `pretrain-generate` run:

~~~bash
HF_REPO=tohio/slm-synthetic-pretrain make pretrain-push
~~~

If you need to publish an older pretraining run, point `configs/synthetic.yaml` at that run first rather than assuming `PRETRAIN_REPORT_RUN` changes the push source.

The publish path verifies the final semantic-quality accepted-token report before uploading `deduped/pretrain.jsonl`.

---

## Generic SFT

### Smoke

~~~bash
make sft-smoke
~~~

Smoke defaults to one family, `grounded_qa_and_reading`, with:

~~~text
1 seed × 1 derivation/seed × 2 tasks/derivation
~~~

Override the smoke family:

~~~bash
SFT_FAMILIES=programming make sft-smoke
~~~

### Production

~~~bash
SFT_GENERATION_RUN=sft-production-001 make sft-generate
~~~

Production defaults:

~~~text
all families
1 seed × 30 derivations/seed × 15 tasks/derivation
~~~

Important variables:

| Variable | Default |
|---|---|
| `SFT_FAMILIES` | `all` |
| `SFT_SEEDS` | `1` |
| `SFT_DERIVATIONS_PER_SEED` | `30` |
| `SFT_TASKS_PER_DERIVATION` | `15` |
| `SFT_ANSWER_BATCH_SIZE` | `4` |
| `SFT_JUDGE_BATCH_SIZE` | `10` |
| `SFT_REVIEWER_BATCH_SIZE` | `10` |
| `SFT_CONCURRENCY` | `8` |
| `SFT_CARDINALITY_FILL_ATTEMPTS` | `3` |
| `SFT_STAGE_BATCH_ATTEMPTS` | `3` |
| `SFT_JACCARD_THRESHOLD` | `0.82` |
| `SFT_SEQUENCE_THRESHOLD` | `0.90` |

Model-role defaults:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| answer | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

Override roles independently:

~~~bash
SFT_DERIVATION_MODEL=<model> \
SFT_TASK_MODEL=<model> \
SFT_ANSWER_MODEL=<model> \
SFT_JUDGE_MODEL=<model> \
SFT_REVIEWER_MODEL=<model> \
make sft-generate
~~~

### Inspect / Report / Push

~~~bash
SFT_INSPECT_RUN=sft-production-001 make sft-inspect
SFT_REPORT_RUN=sft-production-001 make sft-report
SFT_PUSH_RUN=sft-production-001 make sft-push
~~~

Default repository: `tohio/slm-synthetic-sft` through `HF_NAMESPACE=tohio`.

---

## Generic DPO

### Smoke

~~~bash
make dpo-smoke
~~~

Smoke defaults to `instruction_adherence` and:

~~~text
1 seed × 2 derivations/seed × 2 tasks/derivation
~~~

Override the dimension:

~~~bash
DPO_PREFERENCE_DIMENSIONS=factual_accuracy make dpo-smoke
~~~

### Production

~~~bash
DPO_GENERATION_RUN=dpo-production-001 make dpo-generate
~~~

Key planning variables are `DPO_GENERATION_RUN`, `DPO_PREFERENCE_DIMENSIONS`, `DPO_SEEDS`, `DPO_DERIVATIONS_PER_SEED`, `DPO_TASKS_PER_DERIVATION`, and `DPO_PAIR_BATCH_SIZE`. See [Parameter Reference](PARAMETERS.md#generic-dpo-parameters) for their defaults and semantics.

Production defaults to all ten dimensions with:

~~~text
1 seed × 30 derivations/seed × 15 tasks/derivation
~~~

Important variables:

| Variable | Default |
|---|---|
| `DPO_PREFERENCE_DIMENSIONS` | `all` |
| `DPO_SEEDS` | `1` |
| `DPO_DERIVATIONS_PER_SEED` | `30` |
| `DPO_TASKS_PER_DERIVATION` | `15` |
| `DPO_PAIR_BATCH_SIZE` | `4` |
| `DPO_JUDGE_BATCH_SIZE` | `10` |
| `DPO_REVIEWER_BATCH_SIZE` | `10` |
| `DPO_CONCURRENCY` | `8` |
| `DPO_CARDINALITY_FILL_ATTEMPTS` | `3` |
| `DPO_STAGE_BATCH_ATTEMPTS` | `3` |
| `DPO_JACCARD_THRESHOLD` | `0.82` |
| `DPO_SEQUENCE_THRESHOLD` | `0.90` |
| `DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` |

Model-role defaults match generic SFT except the answer role is a pair role:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| pair | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

### Inspect / Report / Push

~~~bash
DPO_INSPECT_RUN=dpo-production-001 make dpo-inspect
DPO_REPORT_RUN=dpo-production-001 make dpo-report
DPO_PUSH_RUN=dpo-production-001 make dpo-push
~~~

Default repository: `tohio/slm-synthetic-dpo`.

---

## Distillation SFT

### Smoke

~~~bash
make distillation-sft-smoke
~~~

Smoke defaults to signal `debugging` and:

~~~text
1 seed × 1 derivation/seed × 2 tasks/derivation
~~~

Override the signal:

~~~bash
DISTILLATION_SFT_SIGNALS=code make distillation-sft-smoke
~~~

### Production

~~~bash
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
make distillation-sft-generate
~~~

Key planning variables are `DISTILLATION_SFT_SIGNALS`, `DISTILLATION_SFT_SEEDS`, `DISTILLATION_SFT_DERIVATIONS_PER_SEED`, `DISTILLATION_SFT_TASKS_PER_DERIVATION`, and `DISTILLATION_SFT_ANSWER_BATCH_SIZE`. See [Parameter Reference](PARAMETERS.md#distillation-sft-parameters).

Production defaults to all ten signals with:

~~~text
1 seed × 30 derivations/seed × 15 tasks/derivation
~~~

Model defaults:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| teacher response | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

Novelty defaults: Jaccard `0.82`, sequence similarity `0.90`.

### Inspect / Report / Push

~~~bash
DISTILLATION_SFT_INSPECT_RUN=distillation-sft-production-001 make distillation-sft-inspect
DISTILLATION_SFT_REPORT_RUN=distillation-sft-production-001 make distillation-sft-report
DISTILLATION_SFT_PUSH_RUN=distillation-sft-production-001 make distillation-sft-push
~~~

Default repository: `tohio/slm-synthetic-distillation-sft`.

---

## Distillation DPO

### Smoke

~~~bash
make distillation-dpo-smoke
~~~

Smoke defaults to dimension `factual_accuracy` and:

~~~text
1 seed × 1 derivation/seed × 2 tasks/derivation
~~~

### Production

~~~bash
DISTILLATION_DPO_TARGET_RUN=distillation-dpo-production-001 \
make distillation-dpo-generate
~~~

Key planning variables are `DISTILLATION_DPO_DIMENSIONS`, `DISTILLATION_DPO_SEEDS`, `DISTILLATION_DPO_DERIVATIONS_PER_SEED`, `DISTILLATION_DPO_TASKS_PER_DERIVATION`, and `DISTILLATION_DPO_PAIR_BATCH_SIZE`. See [Parameter Reference](PARAMETERS.md#distillation-dpo-parameters).

Production defaults to all ten preference dimensions with:

~~~text
1 seed × 30 derivations/seed × 15 tasks/derivation
~~~

Model defaults differ intentionally from generic DPO:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| pair | `deepseek/deepseek-v4-flash` |
| judge | `google/gemma-4-31b-it` |
| reviewer | `openai/gpt-5.6-luna-pro` |

The judge uses the five-gate Distillation-DPO contract; do not substitute generic-DPO judge semantics.

### Inspect / Report / Push

~~~bash
DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-production-001 make distillation-dpo-inspect
DISTILLATION_DPO_REPORT_RUN=distillation-dpo-production-001 make distillation-dpo-report
DISTILLATION_DPO_PUSH_RUN=distillation-dpo-production-001 make distillation-dpo-push
~~~

Default repository: `tohio/slm-synthetic-distillation-dpo`.

---

## Testing and Cleanup

~~~bash
make test
make clean
~~~

`clean` removes generated run directories under the repository's configured data roots. It does not delete Hugging Face repositories.

## See Also

- [Generation Workflow](GENERATION_WORKFLOW.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Generation Families](GENERATION_FAMILIES.md)
