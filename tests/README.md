# Tests

Purpose and usage notes for the repository test suite.

## Purpose

The test suite covers pretraining generation, SFT, DPO, distillation SFT, distillation DPO, taxonomy, holdout behavior, manifests, reports, CLI contracts, telemetry, and publishing boundaries.

## Run Tests

Install the test dependency if needed:

```bash
python -m pip install pytest
```

Run the full suite:

```bash
make test
```

Run focused suites while changing one surface:

```bash
pytest -q tests/test_sft_*.py
pytest -q tests/test_dpo_*.py
pytest -q tests/test_distillation_*.py
pytest -q tests/test_pretrain_*.py tests/test_grounded_*.py
```

## Run-Ladder Checks

Live generation requires `OPENROUTER_API_KEY`. Start with smoke jobs:

```bash
make pretrain-smoke
make sft-smoke
make dpo-smoke
make distillation-sft-smoke
make distillation-dpo-smoke
```

Then inspect public files and manifests:

```bash
make pretrain-inspect
make sft-inspect
make dpo-inspect
make distillation-sft-inspect
make distillation-dpo-inspect
```

Small-scale target overrides should be validated before any full production target.

## Public Row Boundaries

| Surface | Public fields |
|---|---|
| SFT | `id`, `messages`, `metadata` |
| DPO | `id`, `prompt`, `chosen`, `rejected`, `metadata` |
| Distillation SFT | `id`, `prompt`, `reasoning`, `response` |
| Distillation DPO | `id`, `prompt`, `chosen`, `rejected`, `metadata` |

Teacher, provider, run, cost, retry, routing, and internal prompt-spec details belong in local manifests, not public dataset rows.
