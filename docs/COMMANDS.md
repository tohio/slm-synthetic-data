# Command Reference

This repo exposes one Make surface for dataset generation:

| Workflow | Smoke Run | Target Run | Inspect |
|---|---|---|---|
| Pretraining | `make pretrain-smoke` | `make pretrain-generate` | `make pretrain-inspect` |
| Response distillation | `make distill-smoke` | `make distill-generate` | `make distill-inspect` |
| SFT | `make sft-smoke` | `make sft-generate` | `make sft-inspect` |
| DPO | `make dpo-smoke` | `make dpo-generate` | `make dpo-inspect` |

Run `make help` to print the same command surface from the Makefile.

Generation commands use consistent throughput controls:

| Setting | Meaning |
|---|---|
| `*_CONCURRENCY` | Maximum simultaneous provider requests. Adaptive request admission may run below this cap when the provider throttles. |
| `*_BATCH_SIZE` | Maximum rows, prompts, or specs per provider request. Adaptive batch sizing halves failing batches and slowly increases after successful batches. |

## Shared Variables

| Variable | Default | Purpose |
|---|---:|---|
| `PYTHON` | `python` | Python executable. |
| `MODEL` | `openai/gpt-4.1-mini` | Default OpenRouter model for live generation. |
| `MAX_TOKENS` | `4096` | Shared token default for commands that use it. |
| `HF_REPO` | unset | Optional Hugging Face destination for pretraining push. |

## Pretraining

Generate a small run:

```bash
make pretrain-smoke
make pretrain-inspect
```

Generate target data:

```bash
make pretrain-generate \
  PRETRAIN_TARGET_TOKENS=1000000 \
  PRETRAIN_TARGET_CONCURRENCY=4

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `PRETRAIN_RUN` | `pretrain-smoke-001` | Smoke run id. |
| `PRETRAIN_TARGET_RUN` | `pretrain-target-001` | Target run id. |
| `PRETRAIN_TOKENS` | `100000` | Smoke token target. |
| `PRETRAIN_TARGET_TOKENS` | `1000000` | Target token target. |
| `PRETRAIN_BATCH_SIZE` | `32` | Maximum rows per provider request. |
| `PRETRAIN_CONCURRENCY` | `1` | Smoke request concurrency. |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | Target request concurrency. |
| `PRETRAIN_INSPECT_RUN` | `$(PRETRAIN_REPORT_RUN)` | Run inspected by `pretrain-inspect`. |
| `PRETRAIN_MODEL` | `$(MODEL)` | Pretraining model. |
| `PRETRAIN_SIGNAL` | unset | Optional single-signal filter. |

Reports:

```bash
make pretrain-report PRETRAIN_REPORT_RUN=<run-id>
```

Push:

```bash
make pretrain-push HF_REPO=<namespace>/<repo>
```

## Response Distillation

Generate a small run:

```bash
make distill-smoke
make distill-inspect
```

Generate target data:

```bash
make distill-generate \
  DISTILL_TARGET_SIZE=pilot \
  DISTILL_TARGET_RUN=distill-pilot-001

make distill-inspect DISTILL_INSPECT_RUN=distill-pilot-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILL_RUN` | `distill-smoke-001` | Smoke run id. |
| `DISTILL_TARGET_RUN` | `distill-target-001` | Target run id. |
| `DISTILL_TARGET_SIZE` | `pilot` | Target preset for `distill-generate`. |
| `DISTILL_SMOKE_COUNT_PER_SIGNAL` | `2` | Smoke rows per signal. |
| `DISTILL_BATCH_SIZE` | `5` | Maximum prompts per teacher request. |
| `DISTILL_CONCURRENCY` | `1` | Parallel teacher requests. |
| `DISTILL_RUN_ROOT` | `data/distillation/runs` | Run output root. |
| `DISTILL_REPORT_RUN` | `$(DISTILL_RUN)` | Run used by `distill-report`. |
| `DISTILL_INSPECT_RUN` | `$(DISTILL_REPORT_RUN)` | Run inspected by `distill-inspect`. |
| `DISTILL_SIGNALS` | unset | Optional signal list. |
| `DISTILL_MODEL` | `$(MODEL)` | Teacher model. |
| `DISTILL_MAX_TOKENS` | `4096` | Teacher response max tokens. |

Reports:

```bash
make distill-report DISTILL_REPORT_RUN=<run-id>
```

## SFT

Generate a small run:

```bash
make sft-smoke
make sft-inspect
```

Generate target data:

```bash
make sft-generate \
  SFT_FAMILIES=all \
  SFT_TARGET_RUN=sft-target-001 \
  SFT_COUNT_PER_FAMILY=1000 \
  SFT_CONCURRENCY=2

make sft-inspect SFT_INSPECT_RUN=sft-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `SFT_RUN` | `sft-smoke-001` | Smoke run id. |
| `SFT_TARGET_RUN` | `sft-target-001` | Target run id. |
| `SFT_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list. |
| `SFT_FAMILIES` | `all` | Target family list. |
| `SFT_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family. |
| `SFT_COUNT_PER_FAMILY` | `1000` | Target rows per family. |
| `SFT_BATCH_SIZE` | `5` | Maximum specs per teacher request. |
| `SFT_CONCURRENCY` | `1` | Parallel teacher requests. |
| `SFT_RUN_ROOT` | `data/sft/runs` | Run output root. |
| `SFT_REPORT_RUN` | `$(SFT_RUN)` | Run used by `sft-report`. |
| `SFT_INSPECT_RUN` | `$(SFT_REPORT_RUN)` | Run inspected by `sft-inspect`. |
| `SFT_MODEL` | `$(MODEL)` | Teacher model. |
| `SFT_MAX_TOKENS` | `4096` | Teacher response max tokens. |

Reports:

```bash
make sft-report SFT_REPORT_RUN=<run-id>
```

## DPO

Generate a small run:

```bash
make dpo-smoke
make dpo-inspect
```

Generate target data:

```bash
make dpo-generate \
  DPO_FAMILIES=all \
  DPO_TARGET_RUN=dpo-target-001 \
  DPO_COUNT_PER_FAMILY=1000 \
  DPO_CONCURRENCY=2

make dpo-inspect DPO_INSPECT_RUN=dpo-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DPO_RUN` | `dpo-smoke-001` | Smoke run id. |
| `DPO_TARGET_RUN` | `dpo-target-001` | Target run id. |
| `DPO_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list. |
| `DPO_FAMILIES` | `all` | Target family list. |
| `DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family. |
| `DPO_COUNT_PER_FAMILY` | `1000` | Target rows per family. |
| `DPO_BATCH_SIZE` | `5` | Maximum specs per teacher request. |
| `DPO_CONCURRENCY` | `1` | Parallel teacher requests. |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Run output root. |
| `DPO_REPORT_RUN` | `$(DPO_RUN)` | Run used by `dpo-report`. |
| `DPO_INSPECT_RUN` | `$(DPO_REPORT_RUN)` | Run inspected by `dpo-inspect`. |
| `DPO_MODEL` | `$(MODEL)` | Teacher model. |
| `DPO_MAX_TOKENS` | `4096` | Teacher response max tokens. |

Reports:

```bash
make dpo-report DPO_REPORT_RUN=<run-id>
```

## Maintenance

```bash
make test
make clean
```
