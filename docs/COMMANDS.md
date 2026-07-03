# Command Reference

This repository supports four workflows:

| Workflow | Main command family |
|---|---|
| Pretraining synthetic data | `make configure`, `make generate`, `make validate`, `make dedup` |
| Response distillation | `make distill-*` |
| SFT | `make sft-*` |
| DPO | `make dpo-*` |

## Environment

Create `.env` with:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

`HF_TOKEN` is required only when pushing data. Do not commit `.env`.

## Pretraining Variables

| Variable | Default | Description |
|---|---:|---|
| `PROFILE` | `balanced` | Runtime posture: `speed`, `balanced`, or `quality`. |
| `TOKENS` | `200000` | Estimated generation target used to derive row counts. |
| `MODEL` | config default | Optional OpenRouter renderer override. |
| `BATCH` | `32` | Grounded artifacts per request. |
| `CONCURRENCY` | profile default | Number of in-flight batch requests. |
| `RUN` | generated | Optional run name. |
| `HF_REPO` | environment/default | Optional Hugging Face destination override. |
| `SIGNAL` | unset | Restrict a command to one pretraining signal. |
| `STAGE` | `deduped` | Stage for duplicate and length reporting. |
| `PRETRAIN_MANIFEST` | unset | Optional pretraining run-manifest output path. |
| `PRETRAIN_COVERAGE_REPORT` | unset | Optional pretraining coverage-report output path. |
| `PRETRAIN_GENERATION_RUN` | unset | Optional generation-run id written to pretraining reports. |

## Pretraining Commands

Configure a smoke run:

```bash
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_smoke
```

Configure the locked production target:

```bash
make production-config CONCURRENCY=4
```

Run the pretraining pipeline:

```bash
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup
make pretrain-generate-manifest
make pretrain-report-coverage
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped
```

Run one pretraining signal:

```bash
make generate SIGNAL=task_code
```

Publish deduped data:

```bash
make push HF_REPO=<namespace>/<repo>
```

## Distillation Variables

| Variable | Default | Description |
|---|---:|---|
| `DISTILL_SIGNAL` | `arithmetic` | Single signal for prompt, materialize, and batch commands. |
| `DISTILL_SIGNALS` | unset | Space-separated signal list for planning or seed runs. |
| `DISTILL_COUNT` | `2` | Prompt count for `distill-build-prompts`. |
| `DISTILL_START_INDEX` | `1` | First generated prompt index. |
| `DISTILL_TARGET` | `smoke` | Token target preset or explicit target accepted by the planner. |
| `DISTILL_ESTIMATED_TOKENS_PER_ROW` | `512` | Planning estimate used to convert token targets into row counts. |
| `DISTILL_PROMPTS` | `/tmp/<signal>.prompts.jsonl` | Local prompt-record JSONL path. |
| `DISTILL_TEACHER_PROMPT` | `/tmp/<signal>.teacher_prompt.txt` | Rendered teacher prompt path. |
| `DISTILL_TEACHER_RESPONSE` | `/tmp/<signal>.teacher_response.json` | Local teacher response JSON path for non-network materialization. |
| `DISTILL_OUTPUT_DIR` | `data/distillation/datasets` | Public JSONL output directory. |
| `DISTILL_MANIFEST_DIR` | `data/distillation/manifests` | Local manifest output directory. |
| `DISTILL_TEACHER_MODEL` | unset | OpenRouter teacher model. Required for generation/materialization. |
| `DISTILL_GENERATION_RUN` | `smoke-001` | Run id written to manifests. |
| `DISTILL_MAX_TOKENS` | `1024` | Teacher response max tokens. |
| `DISTILL_TOKEN_TARGET` | unset | Optional token-target label for single-batch commands. |
| `DISTILL_RUN_MANIFEST` | `data/distillation/manifests/<run>.manifest.json` | Run manifest used for reports and dataset-card generation. |
| `DISTILL_COVERAGE_REPORT` | `data/distillation/coverage.json` | Coverage-report output path. |
| `DISTILL_DATASET_CARD` | `data/distillation/README.md` | Dataset card output path. |
| `DISTILL_DATASET_NAME` | `SLM Synthetic Distillation` | Dataset card title. |
| `DISTILL_LICENSE` | unset | Optional dataset-card license metadata. |

## Distillation Commands

Plan a token target:

```bash
make distill-plan DISTILL_TARGET=smoke
```

Build seed prompts for one signal:

```bash
make distill-build-prompts DISTILL_SIGNAL=arithmetic DISTILL_COUNT=2
```

Render a teacher prompt without calling the provider:

```bash
make distill-render-teacher-prompt DISTILL_SIGNAL=arithmetic
```

Materialize a local teacher response:

```bash
make distill-materialize-batch \
  DISTILL_SIGNAL=arithmetic \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_TEACHER_RESPONSE=/tmp/arithmetic.teacher_response.json
```

Generate one live OpenRouter batch:

```bash
make distill-generate-batch \
  DISTILL_SIGNAL=arithmetic \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini
```

Generate a seed run across all supported distillation signals:

```bash
make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=distill-smoke-001 \
  DISTILL_TARGET=smoke
```

Generate selected distillation signals:

```bash
make distill-generate-seed-run \
  DISTILL_SIGNALS="arithmetic code debugging" \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_TARGET=smoke
```

Build a dataset card from a run manifest:

```bash
make distill-build-dataset-card \
  DISTILL_RUN_MANIFEST=data/distillation/manifests/distill-smoke-001.manifest.json \
  DISTILL_DATASET_CARD=data/distillation/README.md \
  DISTILL_DATASET_NAME="SLM Synthetic Distillation Smoke" \
  DISTILL_LICENSE=mit
```

Write a coverage report:

```bash
make distill-report-coverage \
  DISTILL_RUN_MANIFEST=data/distillation/manifests/distill-smoke-001.manifest.json
```

## SFT Variables

| Variable | Default | Description |
|---|---:|---|
| `SFT_FAMILY` | `answer_only_arithmetic` | Deterministic seed family for `sft-materialize-seed`. |
| `SFT_SPEC_FAMILY` | `basic_arithmetic_qa` | LLM task-spec family for `sft-build-specs`. |
| `SFT_FAMILIES` | `all` | Families for run-level seed or LLM generation. |
| `SFT_COUNT` | `100` | Row/spec count for single-family commands. |
| `SFT_COUNT_PER_FAMILY` | `$(SFT_COUNT)` | Row/spec count per family for run commands. |
| `SFT_OUTPUT_DIR` | `data/sft/datasets` | Public SFT JSONL output directory. |
| `SFT_MANIFEST_DIR` | `data/sft/manifests` | Local manifest output directory. |
| `SFT_COVERAGE_REPORT` | `data/sft/coverage.json` | Coverage-report output path. |
| `SFT_GENERATION_RUN` | `sft-smoke-001` | Run id written to manifests. |
| `SFT_START_INDEX` | `1` | First generated row/spec index. |
| `SFT_SPECS` | `/tmp/sft.specs.jsonl` | LLM task-spec JSONL path. |
| `SFT_TEACHER_RESPONSE` | `/tmp/sft.teacher_response.json` | Local teacher response JSON path. |
| `SFT_OUTPUT` | `data/sft/datasets/<family>.jsonl` | Single-batch public JSONL output path. |
| `SFT_MANIFEST` | `data/sft/manifests/<family>.<run>.manifest.json` | Single-batch manifest path. |
| `SFT_TEACHER_MODEL` | unset | OpenRouter teacher model. Required for generation/materialization. |
| `SFT_MAX_TOKENS` | `4096` | Teacher response max tokens. |
| `SFT_BATCH_SIZE` | `5` | Specs per live LLM request in run generation. |
| `SFT_MAX_WORKERS` | `1` | Optional worker concurrency for LLM run generation. |
| `SFT_RUN_MANIFEST_FILENAME` | unset | Optional run-manifest filename override. |

## SFT Commands

Build no-network task specs:

```bash
make sft-build-specs \
  SFT_SPEC_FAMILY=basic_arithmetic_qa \
  SFT_COUNT=2 \
  SFT_SPECS=/tmp/sft.specs.jsonl
```

Materialize saved teacher JSON:

```bash
make sft-materialize-llm-batch \
  SFT_SPECS=/tmp/sft.specs.jsonl \
  SFT_TEACHER_RESPONSE=/tmp/sft.teacher_response.json \
  SFT_OUTPUT=/tmp/sft.jsonl \
  SFT_MANIFEST=/tmp/sft.manifest.json \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-local-001
```

Generate one live batch:

```bash
make sft-generate-llm-batch \
  SFT_SPECS=/tmp/sft.specs.jsonl \
  SFT_OUTPUT=/tmp/sft.jsonl \
  SFT_MANIFEST=/tmp/sft.manifest.json \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-live-001 \
  SFT_MAX_TOKENS=2048
```

Generate a live multi-family run:

```bash
make sft-generate-llm-run \
  SFT_FAMILIES="basic_arithmetic_qa repeat_exact_n_times" \
  SFT_COUNT_PER_FAMILY=25 \
  SFT_BATCH_SIZE=5 \
  SFT_MAX_WORKERS=1 \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-smoke-001
```

Materialize deterministic seed rows:

```bash
make sft-materialize-seed SFT_FAMILY=repeat_exact_n_times SFT_COUNT=25
make sft-materialize-seed-run SFT_FAMILIES="answer_only_arithmetic repeat_exact_n_times" SFT_COUNT_PER_FAMILY=25
```

Write a coverage report:

```bash
make sft-report-coverage
```

## DPO Variables

| Variable | Default | Description |
|---|---:|---|
| `DPO_FAMILY` | `answer_only_arithmetic` | Deterministic seed family for `dpo-materialize-seed`. |
| `DPO_SPEC_FAMILY` | `basic_arithmetic_qa` | LLM task-spec family for `dpo-build-specs`. |
| `DPO_FAMILIES` | `all` | Families for run-level seed or LLM generation. |
| `DPO_COUNT` | `100` | Row/spec count for single-family commands. |
| `DPO_COUNT_PER_FAMILY` | `$(DPO_COUNT)` | Row/spec count per family for run commands. |
| `DPO_OUTPUT_DIR` | `data/dpo/datasets` | Public DPO JSONL output directory. |
| `DPO_MANIFEST_DIR` | `data/dpo/manifests` | Local manifest output directory. |
| `DPO_COVERAGE_REPORT` | `data/dpo/coverage.json` | Coverage-report output path. |
| `DPO_GENERATION_RUN` | `dpo-smoke-001` | Run id written to manifests. |
| `DPO_START_INDEX` | `1` | First generated row/spec index. |
| `DPO_SPECS` | `/tmp/dpo.specs.jsonl` | LLM task-spec JSONL path. |
| `DPO_TEACHER_RESPONSE` | `/tmp/dpo.teacher_response.json` | Local teacher response JSON path. |
| `DPO_OUTPUT` | `data/dpo/datasets/<family>.jsonl` | Single-batch public JSONL output path. |
| `DPO_MANIFEST` | `data/dpo/manifests/<family>.<run>.manifest.json` | Single-batch manifest path. |
| `DPO_TEACHER_MODEL` | unset | OpenRouter teacher model. Required for generation/materialization. |
| `DPO_MAX_TOKENS` | `4096` | Teacher response max tokens. |
| `DPO_BATCH_SIZE` | `5` | Specs per live LLM request in run generation. |
| `DPO_MAX_WORKERS` | `1` | Optional worker concurrency for LLM run generation. |
| `DPO_RUN_MANIFEST_FILENAME` | unset | Optional run-manifest filename override. |

## DPO Commands

Build no-network task specs:

```bash
make dpo-build-specs \
  DPO_SPEC_FAMILY=basic_arithmetic_qa \
  DPO_COUNT=2 \
  DPO_SPECS=/tmp/dpo.specs.jsonl
```

Materialize saved teacher JSON:

```bash
make dpo-materialize-llm-batch \
  DPO_SPECS=/tmp/dpo.specs.jsonl \
  DPO_TEACHER_RESPONSE=/tmp/dpo.teacher_response.json \
  DPO_OUTPUT=/tmp/dpo.jsonl \
  DPO_MANIFEST=/tmp/dpo.manifest.json \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-local-001
```

Generate one live batch:

```bash
make dpo-generate-llm-batch \
  DPO_SPECS=/tmp/dpo.specs.jsonl \
  DPO_OUTPUT=/tmp/dpo.jsonl \
  DPO_MANIFEST=/tmp/dpo.manifest.json \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-live-001 \
  DPO_MAX_TOKENS=2048
```

Generate a live multi-family run:

```bash
make dpo-generate-llm-run \
  DPO_FAMILIES="basic_arithmetic_qa repeat_exact_n_times" \
  DPO_COUNT_PER_FAMILY=25 \
  DPO_BATCH_SIZE=5 \
  DPO_MAX_WORKERS=1 \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-smoke-001
```

Materialize deterministic seed rows:

```bash
make dpo-materialize-seed DPO_FAMILY=repeat_exact_n_times DPO_COUNT=25
make dpo-materialize-seed-run DPO_FAMILIES="answer_only_arithmetic repeat_exact_n_times" DPO_COUNT_PER_FAMILY=25
```

Write a coverage report:

```bash
make dpo-report-coverage
```

## Supported SFT/DPO Families

LLM spec families:

```text
ai_concept_explanation
basic_arithmetic_qa
capital_city_qa
clear_sky_color_qa
code_explanation_no_code
code_expression_result
code_generation_function
direct_division
direct_subtraction
function_completion_body_only
list_exact_n_items
private_or_unverifiable_company_fact
repeat_exact_n_times
short_factual_stop_behavior
```

Deterministic seed families also include `answer_only_arithmetic`.

## Tests

```bash
python -m compileall -q slm_synth tests
pytest -q
```

Focused validation:

```bash
pytest -q \
  tests/test_sft_*.py \
  tests/test_dpo_*.py \
  tests/test_distillation_*.py \
  tests/test_taxonomy.py \
  tests/test_eval_holdouts.py \
  tests/test_pretrain_manifest.py
```
