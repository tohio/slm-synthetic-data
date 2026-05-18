# Tests

Unit tests for the SLM synthetic-data pipeline.

The tests cover core behavior around generation, validation, schema handling, and deduplication.

Run tests with:

```bash
python -m pytest -q
```

or:

```bash
make test
```

---

## Test Areas

```text
tests/test_generate.py    # Generation flow and batch behavior.
tests/test_validate.py    # Validation logic.
tests/test_dedup.py       # Deduplication behavior.
tests/test_schemas.py     # Signal schema checks.
```

---

## Recommended Manual Smoke Tests

The repo also relies on short live-generation smoke tests because model behavior can change.

Small full-pipeline smoke test:

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

Signal-specific smoke test:

```bash
make generate SIGNAL=educational_qa_mcq
```

---

## What to Watch

For synthetic generation, tests passing is not enough. Also check:

- rejected batches
- bad JSON count
- validation rejects
- exact duplicate rate
- dedup retention

Healthy smoke tests should have:

```text
bad_json = 0
rejected_batches near 0
raw duplicate rate ideally below 5%
deduped duplicate rate = 0
```
