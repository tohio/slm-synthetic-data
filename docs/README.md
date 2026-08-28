# Documentation

Reference and operational guides for the five supported synthetic dataset pipelines.

## Architecture and Concepts

- [Architecture](ARCHITECTURE.md) — runtime boundary, dataset ownership, stage architecture, and publication flow.
- [Dataset Purpose and Contracts](DATASET_PURPOSE.md) — what each dataset is for, its public row contract, and its downstream consumer.
- [Generation Families and Dimensions](GENERATION_FAMILIES.md) — pretraining signals, SFT families, DPO dimensions, and distillation coverage.

## Operations

- [Command Reference](COMMANDS.md) — supported Make targets, variables, model overrides, routing, qualification, and examples.
- [Generation Workflow](GENERATION_WORKFLOW.md) — smoke-to-production workflow, stage outputs, inspection, reporting, and publication.
- [Troubleshooting](TROUBLESHOOTING.md) — common provider, structured-output, underfill, holdout, reporting, and publication failures.
- [Disk Setup](DISK_SETUP.md) — optional secondary-volume setup for large local runs.
