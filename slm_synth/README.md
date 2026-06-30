# `slm_synth` Package

## Shared Modules

| Module | Purpose |
|---|---|
| `llm.py` | OpenRouter client, structured JSON output, retries, and adaptive concurrency. |
| `paths.py` | Shared data-path helpers. |
| `rate_limit.py` | Request pacing helpers. |
| `model_support.py` | Supported provider/model guidance. |

## Pretraining Modules

Implementation lives under `slm_synth.pretrain`.

| Module | Purpose |
|---|---|
| `pretrain/artifacts/` | Deterministic grounded artifacts and preflight quality checks. |
| `pretrain/generate.py` | Pretraining generation orchestration and resume. |
| `pretrain/grounded.py` | Grounded batch rendering, anchoring, and persistence. |
| `pretrain/validate.py` | Raw-to-validated record validation. |
| `pretrain/dedup.py` | Exact deduplication. |
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
