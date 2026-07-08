# Tests

The test suite covers pretraining generation, response distillation, SFT, DPO, taxonomy, holdout behavior, manifests, reports, and CLI contracts.

## Run Tests

Install the test dependency if needed:

```bash
python -m pip install pytest
```

Run the full suite:

```bash
make test
```

Run a focused suite:

```bash
pytest -q \
  tests/test_sft_*.py \
  tests/test_dpo_*.py \
  tests/test_distillation_*.py \
  tests/test_taxonomy.py \
  tests/test_eval_holdouts.py \
  tests/test_pretrain_manifest.py
```

## Live Smoke Runs

Live generation requires `OPENROUTER_API_KEY`.

```bash
make pretrain-smoke
make distillation-sft-smoke
make sft-smoke
make dpo-smoke
```

Inspect generated files:

```bash
make pretrain-inspect
make distillation-sft-inspect
make sft-inspect
make dpo-inspect
```

## Expected Public Rows

Distillation rows:

```text
id
prompt
reasoning
response
```

SFT rows:

```text
id
messages
metadata
```

DPO rows:

```text
id
prompt
chosen
rejected
metadata
```

Teacher/provider/run/cost/retry details belong in local manifests, not public dataset rows.
