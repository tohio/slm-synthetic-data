# Tests

This directory contains tests for pretraining synthetic generation, response distillation, SFT, DPO, taxonomy, and holdout behavior.

## Test Categories

| Category | Purpose | Network required |
|---|---|---|
| Schema tests | Validate public row shape and field constraints. | No |
| Dedup tests | Verify exact dedup behavior. | No |
| Generate tests | Exercise generator plumbing and parsing behavior. | Usually no |
| Validate tests | Verify raw-to-validated stage behavior. | No |
| Distillation tests | Validate public row schema, teacher matching, manifests, CLI helpers, reports, and card generation. | No for unit tests |
| SFT tests | Validate seed rows, LLM batch contracts, materialization, run manifests, CLI helpers, and coverage reports. | No for unit tests |
| DPO tests | Validate preference rows, LLM batch contracts, materialization, run manifests, CLI helpers, and coverage reports. | No for unit tests |
| Taxonomy and holdout tests | Validate exact prompt and structured holdout protection. | No |

## Run Tests

Install test dependency if needed:

```bash
python -m pip install pytest
```

Run the suite:

```bash
pytest -q
```

Focused suite:

```bash
pytest -q \
  tests/test_sft_*.py \
  tests/test_dpo_*.py \
  tests/test_distillation_*.py \
  tests/test_taxonomy.py \
  tests/test_eval_holdouts.py \
  tests/test_pretrain_manifest.py
```

## Pretraining Smoke Test

```bash
make configure TOKENS=100000 BATCH=32 CONCURRENCY=4 RUN=grounded_smoke
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

Public distillation rows should contain only:

```text
id
prompt
reasoning
response
```

## SFT/DPO Non-Network Smoke Test

```bash
make sft-build-specs \
  SFT_SPEC_FAMILY=basic_arithmetic_qa \
  SFT_COUNT=2 \
  SFT_SPECS=/tmp/sft.specs.jsonl

make dpo-build-specs \
  DPO_SPEC_FAMILY=basic_arithmetic_qa \
  DPO_COUNT=2 \
  DPO_SPECS=/tmp/dpo.specs.jsonl
```

Use saved teacher JSON responses matching the generated spec ids, then materialize:

```bash
make sft-materialize-llm-batch \
  SFT_SPECS=/tmp/sft.specs.jsonl \
  SFT_TEACHER_RESPONSE=/tmp/sft.teacher_response.json \
  SFT_OUTPUT=/tmp/sft.jsonl \
  SFT_MANIFEST=/tmp/sft.manifest.json \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini

make dpo-materialize-llm-batch \
  DPO_SPECS=/tmp/dpo.specs.jsonl \
  DPO_TEACHER_RESPONSE=/tmp/dpo.teacher_response.json \
  DPO_OUTPUT=/tmp/dpo.jsonl \
  DPO_MANIFEST=/tmp/dpo.manifest.json \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini
```

Public SFT rows should contain only:

```text
id
messages
metadata
```

Public DPO rows should contain only:

```text
id
prompt
chosen
rejected
metadata
```

## Live Smoke Tests

Live generation requires `OPENROUTER_API_KEY`.

```bash
make distill-generate-seed-run \
  DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini \
  DISTILL_GENERATION_RUN=distill-smoke-001 \
  DISTILL_TARGET=smoke

make sft-generate-llm-run \
  SFT_FAMILIES=basic_arithmetic_qa \
  SFT_COUNT_PER_FAMILY=2 \
  SFT_BATCH_SIZE=2 \
  SFT_MAX_WORKERS=1 \
  SFT_TEACHER_MODEL=openai/gpt-4.1-mini \
  SFT_GENERATION_RUN=sft-smoke-001 \
  SFT_MAX_TOKENS=2048

make dpo-generate-llm-run \
  DPO_FAMILIES=basic_arithmetic_qa \
  DPO_COUNT_PER_FAMILY=2 \
  DPO_BATCH_SIZE=2 \
  DPO_MAX_WORKERS=1 \
  DPO_TEACHER_MODEL=openai/gpt-4.1-mini \
  DPO_GENERATION_RUN=dpo-smoke-001 \
  DPO_MAX_TOKENS=2048
```
