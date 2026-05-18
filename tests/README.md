# Tests

This directory contains tests for the synthetic data pipeline.

## Test categories

| Category | Purpose | Network required |
|---|---|---|
| Schema tests | Validate record shape and field constraints. | No |
| Dedup tests | Verify exact dedup behavior. | No |
| Generate tests | Exercise generator plumbing and parsing behavior. | Usually no, unless explicitly configured for live calls. |
| Validate tests | Verify raw-to-validated stage behavior. | No |

## Run tests

Install test dependency if needed:

```bash
python -m pip install pytest
```

Run the suite:

```bash
python -m pytest -q
```

## Pipeline smoke test

For an end-to-end local run using the supported Groq path:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

Expected characteristics for a healthy smoke test:

- generation completes with `rejected_batches=0`,
- validation rejects are zero or very low,
- exact duplicate rate before dedup is low,
- deduped duplicate rate is `0.00%`,
- no bad JSON is reported.

## Supported live models

Live generation tests should use one of the validated models:

- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`

Other models may be used experimentally, but test results should not be interpreted as production support.

## Dedup expectations

For synthetic data, exact dedup is expected. Fuzzy dedup should not be enabled by default in tests for these signals.

A large exact-duplicate drop rate usually indicates a generation diversity issue, not a dedup success.
