# Command Reference

Lookup reference for supported Make targets and common variables. For end-to-end run order across all generation surfaces, see `GENERATION_WORKFLOW.md`.

## Command Groups

| Workflow | Smoke Run | Target Run | Inspect | Push |
|---|---|---|---|---|
| Pretraining | `make pretrain-smoke` | `make pretrain-generate` | `make pretrain-inspect` | `make pretrain-push` |
| SFT | `make sft-smoke` | `make sft-generate` | `make sft-inspect` | `make sft-push` |
| DPO | `make dpo-smoke` | `make dpo-generate` | `make dpo-inspect` | `make dpo-push` |
| Distillation SFT | `make distillation-sft-smoke` | `make distillation-sft-generate` | `make distillation-sft-inspect` | `make distillation-sft-push` |
| Distillation DPO | `make distillation-dpo-smoke` | `make distillation-dpo-generate` | `make distillation-dpo-inspect` | `make distillation-dpo-push` |

Run `make help` to print the command surface from the Makefile.

## Shared Live-Generation Variables

| Variable | Default | Purpose |
|---|---:|---|
| `PYTHON` | `python` | Python executable. |
| `MODEL` | `openai/gpt-4.1-mini` | Default OpenRouter model for live generation. |
| `MAX_TOKENS` | `4096` | Shared token default for commands that use it. |
| `OPENROUTER_ROUTING_MODE` | `auto` | Routing policy: `auto`, `prefer`, or `strict`. |
| `OPENROUTER_PROVIDER` | unset | Provider slug used by `prefer` or `strict` routing. |
| `HF_NAMESPACE` | `tohio` | Default Hugging Face namespace for dataset-specific push targets. |
| `HF_REPO` | unset | Explicit Hugging Face destination for push targets that use one repo id. |
| `HF_PRIVATE` | unset | Set to `true`, `yes`, or `1` for private Hugging Face repos. |

OpenRouter routing defaults to `auto`. Use `prefer` to try one provider first while allowing fallback, or `strict` to require one provider. `prefer` and `strict` require `OPENROUTER_PROVIDER`.

## Pretraining

```bash
make pretrain-smoke
make pretrain-inspect
```

```bash
PRETRAIN_TARGET_TOKENS=1000000 PRETRAIN_TARGET_CONCURRENCY=4 make pretrain-generate
make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `PRETRAIN_RUN` | `pretrain-smoke-001` | Smoke run id. |
| `PRETRAIN_TARGET_RUN` | `pretrain-target-001` | Target run id. |
| `PRETRAIN_TOKENS` | `100000` | Smoke token target. |
| `PRETRAIN_TARGET_TOKENS` | `1000000` | Target token target. |
| `PRETRAIN_BATCH_SIZE` | `32` | Maximum rows per provider request. |
| `PRETRAIN_CONCURRENCY` | `1` | Smoke request concurrency. |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | Target request concurrency. |
| `PRETRAIN_MODEL` | `$(MODEL)` | Pretraining model. |
| `PRETRAIN_SIGNAL` | unset | Optional single-signal filter. |

## SFT

```bash
make sft-smoke
make sft-inspect
```

```bash
SFT_TARGET_ROWS=14000 SFT_TARGET_RUN=sft-target-001 make sft-generate
make sft-inspect SFT_INSPECT_RUN=sft-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `SFT_RUN` | `sft-smoke-001` | Smoke run id. |
| `SFT_TARGET_RUN` | `sft-target-001` | Target run id. |
| `SFT_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list. |
| `SFT_FAMILIES` | `all` | Target family list. |
| `SFT_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family. |
| `SFT_TARGET_ROWS` | `14000` | Target rows across selected families. |
| `SFT_COUNT_PER_FAMILY` | `1000` | Explicit lower-level rows-per-family override. |
| `SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request. |
| `SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `SFT_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests. |
| `SFT_RUN_ROOT` | `data/sft/runs` | Run output root. |
| `SFT_MODEL` | `$(MODEL)` | Teacher model. |
| `SFT_MAX_BACKFILL_ROUNDS` | `2` | Accepted-target backfill budget. |

## DPO

```bash
make dpo-smoke
make dpo-inspect
```

```bash
DPO_TARGET_PAIRS=14000 DPO_TARGET_RUN=dpo-target-001 make dpo-generate
make dpo-inspect DPO_INSPECT_RUN=dpo-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `DPO_RUN` | `dpo-smoke-001` | Smoke run id. |
| `DPO_TARGET_RUN` | `dpo-target-001` | Target run id. |
| `DPO_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list. |
| `DPO_FAMILIES` | `all` | Target family list. |
| `DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family. |
| `DPO_TARGET_PAIRS` | `14000` | Target pairs across selected families. |
| `DPO_COUNT_PER_FAMILY` | `1000` | Explicit lower-level pairs-per-family override. |
| `DPO_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request. |
| `DPO_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `DPO_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests. |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Run output root. |
| `DPO_MODEL` | `$(MODEL)` | Teacher model. |
| `DPO_MAX_BACKFILL_ROUNDS` | `2` | Accepted-target backfill budget. |

## Distillation SFT

```bash
make distillation-sft-smoke
make distillation-sft-inspect
```

```bash
DISTILLATION_SFT_TARGET_ROWS=100000 DISTILLATION_SFT_TARGET_RUN=distillation-sft-target-001 make distillation-sft-generate
make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_SFT_RUN` | `distillation-sft-smoke-001` | Smoke run id. |
| `DISTILLATION_SFT_TARGET_RUN` | `distillation-sft-target-001` | Target run id. |
| `DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL` | `2` | Smoke rows per signal. |
| `DISTILLATION_SFT_TARGET_ROWS` | `100000` | Target accepted public rows. |
| `DISTILLATION_SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum prompts per teacher request. |
| `DISTILLATION_SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `DISTILLATION_SFT_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests. |
| `DISTILLATION_SFT_RUN_ROOT` | `data/distillation/runs` | Run output root. |
| `DISTILLATION_SFT_SIGNALS` | unset | Optional signal list. |
| `DISTILLATION_SFT_MODEL` | `$(MODEL)` | Teacher model. |
| `DISTILLATION_SFT_MAX_BACKFILL_ROUNDS` | `2` | Accepted-target backfill budget after response quality gates. |
| `DISTILLATION_SFT_MIN_UNIQUE_RESPONSE_RATIO` | `0.75` | Minimum exact unique-response ratio required for every signal by report and publish gates. |

## Distillation DPO

```bash
make distillation-dpo-smoke
make distillation-dpo-inspect
```

```bash
DISTILLATION_DPO_TARGET_PAIRS=50000 DISTILLATION_DPO_TARGET_RUN=distillation-dpo-target-001 make distillation-dpo-generate
make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_DPO_RUN` | `distillation-dpo-smoke-001` | Smoke run id. |
| `DISTILLATION_DPO_TARGET_RUN` | `distillation-dpo-target-001` | Target run id. |
| `DISTILLATION_DPO_SMOKE_FAMILIES` | `teacher_response_preference` | Smoke family list. |
| `DISTILLATION_DPO_FAMILIES` | `all` | Target family list. |
| `DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family. |
| `DISTILLATION_DPO_TARGET_PAIRS` | `50000` | Target accepted preference pairs. |
| `DISTILLATION_DPO_RUN_ROOT` | `data/distillation-dpo/runs` | Run output root. |
| `DISTILLATION_DPO_MODEL` | `$(MODEL)` | Teacher lineage model recorded in manifests. |
| `DISTILLATION_DPO_MAX_BACKFILL_ROUNDS` | `2` | Accepted-target backfill budget after pair quality gates. |

## Reports and Publishing

```bash
make pretrain-report PRETRAIN_REPORT_RUN=<run-id>
make sft-report SFT_REPORT_RUN=<run-id>
make dpo-report DPO_REPORT_RUN=<run-id>
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=<run-id>
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=<run-id>
```

```bash
make pretrain-push HF_REPO=<namespace>/<repo>
make sft-push SFT_HF_REPO=<namespace>/<repo>
make dpo-push DPO_HF_REPO=<namespace>/<repo>
make distillation-sft-push DISTILLATION_SFT_HF_REPO=<namespace>/<repo>
make distillation-dpo-push DISTILLATION_DPO_HF_NAMESPACE=<namespace>
```

## Hugging Face Dataset Deletion

Deletion targets remove Hugging Face **dataset repositories** only. They do not delete local run data under `data/`.

All deletion targets are dry-run by default. Set `HF_DELETE_YES=1` to actually delete selected repos.

### Dry-run examples

```bash
make hf-delete-distillation
make hf-delete-legacy-distillation-dpo
make hf-delete-sft
make hf-delete-dpo
```

Delete target output should list selected repositories and print `DRY RUN ONLY`.

### Actual deletion examples

```bash
HF_DELETE_YES=1 make hf-delete-distillation
HF_DELETE_YES=1 make hf-delete-legacy-distillation-dpo
```

Delete an exact repo:

```bash
make hf-delete-datasets HF_DELETE_REPO=tohio/slm-synthetic-distillation-dpo
HF_DELETE_YES=1 make hf-delete-datasets HF_DELETE_REPO=tohio/slm-synthetic-distillation-dpo
```

Delete repos from a file:

```bash
make hf-delete-datasets HF_DELETE_REPO_FILE=repos-to-delete.txt
HF_DELETE_YES=1 make hf-delete-datasets HF_DELETE_REPO_FILE=repos-to-delete.txt
```

The repo file should contain one dataset repo id per line:

```text
tohio/slm-synthetic-distillation-sft
tohio/slm-synthetic-distillation-dpo
```

### Delete targets

| Target | Purpose |
|---|---|
| `make hf-delete-datasets` | Delete exact repos from `HF_DELETE_REPO` or `HF_DELETE_REPO_FILE`. |
| `make hf-delete-sft` | Delete `slm-synthetic-sft-*` family dataset repos. |
| `make hf-delete-dpo` | Delete `slm-synthetic-dpo-*` family dataset repos. |
| `make hf-delete-distillation` | Delete `slm-synthetic-distillation-sft` and `slm-synthetic-distillation-dpo`. |
| `make hf-delete-legacy-distillation-dpo` | Delete the old long distillation-DPO repo name. |

### Delete variables

| Variable | Default | Purpose |
|---|---:|---|
| `HF_DELETE_NAMESPACE` | `$(HF_NAMESPACE)` | Hugging Face namespace for generated delete targets. |
| `HF_DELETE_REPO` | unset | Exact dataset repo id to delete. |
| `HF_DELETE_REPO_FILE` | unset | File containing one exact dataset repo id per line. |
| `HF_DELETE_YES` | unset | Set to `1`, `true`, or `yes` to actually delete. |
| `SFT_HF_PREFIX` | `slm-synthetic-sft` | Prefix used by `make hf-delete-sft`. |
| `DPO_HF_PREFIX` | `slm-synthetic-dpo` | Prefix used by `make hf-delete-dpo`. |

Actual deletion requires `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` in the environment.

## Maintenance

```bash
make test
make clean
```
