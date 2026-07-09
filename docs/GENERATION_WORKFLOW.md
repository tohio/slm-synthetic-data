# Generation Workflow

End-to-end run ladder for generating synthetic datasets safely.

## Run Ladder

Use the same order for every active generation surface:

1. Run a smoke job.
2. Inspect public rows or pairs.
3. Inspect run manifests, planning fields, telemetry, and public-directory hygiene.
4. Run a small target override.
5. Inspect the small-scale output.
6. Run the full target only after earlier outputs pass inspection.
7. Push only public artifacts.

Public dataset directories should contain final public files only. Batch shards, partial files, rejected rows, retry files, provider internals, and scratch files stay out of public upload discovery.

## Generic SFT

Smoke:

```bash
make sft-smoke
make sft-inspect SFT_INSPECT_RUN=sft-smoke-001
```

Small target override:

```bash
SFT_TARGET_ROWS=100 SFT_TARGET_RUN=sft-small-001 make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-small-001
make sft-report SFT_REPORT_RUN=sft-small-001
```

Production target:

```bash
SFT_TARGET_ROWS=14000 SFT_TARGET_RUN=sft-prod-001 make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-prod-001
make sft-report SFT_REPORT_RUN=sft-prod-001
```

Push after inspection:

```bash
make sft-push SFT_PUSH_RUN=sft-prod-001
```

Public rows are written under `data/sft/runs/<run>/datasets/`.

## Generic DPO

Smoke:

```bash
make dpo-smoke
make dpo-inspect DPO_INSPECT_RUN=dpo-smoke-001
```

Small target override:

```bash
DPO_TARGET_PAIRS=100 DPO_TARGET_RUN=dpo-small-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-small-001
make dpo-report DPO_REPORT_RUN=dpo-small-001
```

Production target:

```bash
DPO_TARGET_PAIRS=14000 DPO_TARGET_RUN=dpo-prod-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-prod-001
make dpo-report DPO_REPORT_RUN=dpo-prod-001
```

Push after inspection:

```bash
make dpo-push DPO_PUSH_RUN=dpo-prod-001
```

Public pairs are written under `data/dpo/runs/<run>/datasets/`.

## Distillation SFT

Smoke:

```bash
make distillation-sft-smoke
make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-smoke-001
```

Small target override:

```bash
DISTILLATION_SFT_TARGET_ROWS=100 DISTILLATION_SFT_TARGET_RUN=distillation-sft-small-001 make distillation-sft-generate

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-small-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-small-001
```

Production target:

```bash
DISTILLATION_SFT_TARGET_ROWS=100000 DISTILLATION_SFT_TARGET_RUN=distillation-sft-prod-001 make distillation-sft-generate

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-prod-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-prod-001
```

Push after inspection:

```bash
make distillation-sft-push DISTILLATION_SFT_PUSH_RUN=distillation-sft-prod-001
```

Public rows are written under `data/distillation/runs/<run>/datasets/` as per-signal JSONL files.

## Distillation DPO

Smoke:

```bash
make distillation-dpo-smoke
make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-smoke-001
```

Small target override:

```bash
DISTILLATION_DPO_TARGET_PAIRS=100 DISTILLATION_DPO_TARGET_RUN=distillation-dpo-small-001 make distillation-dpo-generate

make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-small-001
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=distillation-dpo-small-001
```

Production target:

```bash
DISTILLATION_DPO_TARGET_PAIRS=50000 DISTILLATION_DPO_TARGET_RUN=distillation-dpo-prod-001 make distillation-dpo-generate

make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-prod-001
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=distillation-dpo-prod-001
```

Push after inspection:

```bash
make distillation-dpo-push DISTILLATION_DPO_PUSH_RUN=distillation-dpo-prod-001
```

Public pairs are written under `data/distillation-dpo/runs/<run>/datasets/`.

## Pretraining

Smoke:

```bash
make pretrain-smoke
make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-smoke-001
```

Small target override:

```bash
PRETRAIN_TARGET_TOKENS=100000 PRETRAIN_TARGET_RUN=pretrain-small-001 make pretrain-generate

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-small-001
make pretrain-report PRETRAIN_REPORT_RUN=pretrain-small-001
```

Production target:

```bash
PRETRAIN_TARGET_TOKENS=1000000 PRETRAIN_TARGET_RUN=pretrain-prod-001 make pretrain-generate

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-prod-001
make pretrain-report PRETRAIN_REPORT_RUN=pretrain-prod-001
```

Push after inspection:

```bash
make pretrain-push HF_REPO=<namespace>/<repo>
```

Grounded pretraining records are written under `data/runs/<run>/deduped/`.

## Validation Checklist

For each run, inspect:

- public rows or pairs for schema, formatting, and obvious quality failures
- run manifest planning fields and telemetry
- retry counts, adaptive batch failures, request tokens, and aggregate request seconds
- public dataset directory hygiene
- coverage reports or dataset cards before publishing

## See Also

- `COMMANDS.md` for Make target reference.
- `DATASET_PURPOSE.md` for artifact families and public row contracts.
