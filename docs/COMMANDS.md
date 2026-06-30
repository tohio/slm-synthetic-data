# Command Reference

This repository supports two workflows:

| Workflow | Main command family |
|---|---|
| Pretraining synthetic data | `make configure`, `make generate`, `make validate`, `make dedup` |
| Response distillation data | `make distill-*` |

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
| `DISTILL_RUN_MANIFEST` | `data/distillation/manifests/<run>.manifest.json` | Run manifest used for dataset-card generation. |
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
  DISTILL_GENERATION_RUN=smoke-001 \
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
  DISTILL_RUN_MANIFEST=data/distillation/manifests/smoke-001.manifest.json \
  DISTILL_DATASET_CARD=data/distillation/README.md \
  DISTILL_DATASET_NAME="SLM Synthetic Distillation Smoke" \
  DISTILL_LICENSE=mit
```

## Tests

```bash
pytest -q
```

Focused distillation tests:

```bash
pytest -q tests/test_distillation_*.py
```
