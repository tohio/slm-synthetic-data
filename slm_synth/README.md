# `slm_synth` Package

## Shared Modules

| Module | Purpose |
|---|---|
| `llm.py` | OpenRouter client, structured JSON output, retries, and adaptive concurrency. |
| `paths.py` | Shared data-path helpers. |
| `rate_limit.py` | Request pacing helpers. |
| `model_support.py` | Supported provider/model guidance. |
| `taxonomy/` | Training labels, eval-family labels, and holdout registry. |

OpenRouter is the only supported production provider.

## Pretraining Modules

Implementation lives under `slm_synth.pretrain`.

| Module | Purpose |
|---|---|
| `pretrain/artifacts/` | Deterministic grounded artifacts and preflight quality checks. |
| `pretrain/generate.py` | Pretraining generation orchestration and resume. |
| `pretrain/grounded.py` | Grounded batch rendering, anchoring, and persistence. |
| `pretrain/validate.py` | Raw-to-validated record validation. |
| `pretrain/dedup.py` | Exact deduplication. |
| `pretrain/manifest.py` | Run manifest and coverage-report generation. |
| `pretrain/report_artifacts.py` | Artifact duplicate, family, and quality reporting. |
| `pretrain/report_lengths.py` | Per-record size estimation for row-target calibration. |
| `pretrain/push_hf.py` | Hugging Face publishing. |

Compatibility wrappers remain at old module paths such as `slm_synth.generate`, `slm_synth.validate`, and `slm_synth.dedup`.

## Distillation Modules

Implementation lives under `slm_synth.distillation`.

| Module | Purpose |
|---|---|
| `distillation/schema.py` | Public row and teacher-output validation. |
| `distillation/signals.py` | Supported distillation signal names. |
| `distillation/prompts.py` | Local prompt-record validation. |
| `distillation/seeds.py` | Built-in seed prompts. |
| `distillation/batches.py` | Teacher batch prompt and response formatting. |
| `distillation/generation.py` | OpenRouter teacher generation for one signal batch. |
| `distillation/orchestration.py` | Multi-signal seed-run orchestration. |
| `distillation/io.py` | Public JSONL, per-signal manifest, and run-manifest writers. |
| `distillation/budget.py` | Token-target planning. |
| `distillation/card.py` | Dataset-card generation from run manifests. |
| `distillation/cli.py` | Command-line entrypoint for distillation helpers. |

## SFT Modules

Implementation lives under `slm_synth.sft`.

| Module | Purpose |
|---|---|
| `sft/schema.py` | Public SFT row validation. |
| `sft/specs.py` | LLM task-spec validation and teacher-visible spec shaping. |
| `sft/spec_builders.py` | Scalable task-spec builders for eval-shaped families. |
| `sft/batches.py` | SFT batch prompt and teacher response contracts. |
| `sft/generation.py` | Local materialization and OpenRouter batch generation. |
| `sft/runs.py` | Multi-family seed and LLM run orchestration. |
| `sft/seeds.py` | Deterministic seed rows. |
| `sft/manifest.py` | Dataset and run manifests. |
| `sft/report.py` | Coverage reports. |
| `sft/cli.py` | Command-line entrypoint for SFT helpers. |

## DPO Modules

Implementation lives under `slm_synth.dpo`.

| Module | Purpose |
|---|---|
| `dpo/schema.py` | Public DPO row validation. |
| `dpo/specs.py` | LLM task-spec validation and teacher-visible spec shaping. |
| `dpo/spec_builders.py` | Scalable task-spec builders for eval-shaped families. |
| `dpo/batches.py` | DPO batch prompt and teacher response contracts. |
| `dpo/generation.py` | Local materialization and OpenRouter batch generation. |
| `dpo/runs.py` | Multi-family seed and LLM run orchestration. |
| `dpo/seeds.py` | Deterministic seed rows. |
| `dpo/manifest.py` | Dataset and run manifests. |
| `dpo/report.py` | Coverage reports. |
| `dpo/cli.py` | Command-line entrypoint for DPO helpers. |
