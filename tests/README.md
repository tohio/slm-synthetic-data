# Tests

This directory contains tests for pretraining synthetic generation and response distillation.

## Test Categories

| Category | Purpose | Network required |
|---|---|---|
| Schema tests | Validate record shape and field constraints. | No |
| Dedup tests | Verify exact dedup behavior. | No |
| Generate tests | Exercise generator plumbing and parsing behavior. | Usually no |
| Validate tests | Verify raw-to-validated stage behavior. | No |
| Distillation tests | Validate public row schema, teacher matching, manifests, CLI helpers, and card generation. | No for unit tests |

## Run Tests

Install test dependency if needed:

```bash
python -m pip install pytest
```

Run the suite:

```bash
pytest -q
```

Focused distillation suite:

```bash
pytest -q tests/test_distillation_*.py
```

## Pretraining Smoke Test

```bash
make configure TOKENS=100000 BATCH=32 CONCURRENCY=4 RUN=grounded_smoke
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped
```

Expected characteristics:

- generation completes without unresolved failed batches,
- validation rejects are zero or low enough to inspect,
- exact duplicate rate is low before dedup,
- deduped duplicate rate is `0.00%`,
- no bad JSON is reported.

## Distillation Non-Network Smoke Test

```bash
make distill-build-prompts DISTILL_SIGNAL=arithmetic DISTILL_COUNT=2
make distill-render-teacher-prompt DISTILL_SIGNAL=arithmetic
```

Create a local teacher response JSON matching the rendered prompt ids, then run:

```bash
make distill-materialize-batch \
  DISTILL_SIGNAL=arithmetic \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_TEACHER_RESPONSE=/tmp/arithmetic.teacher_response.json
```

## Distillation Live Smoke Test

Requires `OPENROUTER_API_KEY`:

```bash
make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=smoke-001 \
  DISTILL_TARGET=smoke
```

Public distillation rows should contain only:

```text
id
prompt
reasoning
response
```
