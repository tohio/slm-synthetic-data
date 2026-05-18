# Prompts

This directory contains Python prompt templates for the synthetic generation signals.

The current prompt system uses Python modules instead of YAML prompt files:

```text
prompts/
├── wrapper.py
├── arithmetic.py
├── task_code.py
├── educational_qa_mcq.py
└── factual_restraint.py
```

---

## JSON Contract

The wrapper asks the model to return a JSON object with an `items` array:

```json
{
  "items": [
    { "type": "..." }
  ]
}
```

This is more reliable with Groq JSON object mode than asking for a top-level JSON array.

Do not change the prompt back to a bare array without also updating parsing and tests.

---

## Diversity Context

Generation uses per-batch diversity context from:

```text
slm_synth/diversity.py
```

The diversity context adds signal-specific variation such as:

- batch nonce / seed
- topic buckets
- difficulty levels
- stem patterns
- number ranges
- task-code categories
- MCQ subject and scenario rotation
- factual-restraint uncertainty styles

This is important for scaling. Without diversity controls, exact duplicate rates can become very high even when JSON parsing succeeds.

---

## Signal Templates

### `arithmetic.py`

Generates integer arithmetic and simple numeric reasoning. The prompt should preserve variation across:

- operation type
- number range
- problem format
- word-problem context
- missing-value or comparison style

### `task_code.py`

Generates Python task/code records. The prompt should keep code short and JSON-safe:

- no markdown fences
- no triple backticks
- escaped newlines inside JSON strings
- concise plans
- beginner/intermediate Python tasks

### `educational_qa_mcq.py`

Generates educational multiple-choice questions. This prompt has stronger diversity requirements because generic MCQs can duplicate easily. It rotates:

- subject
- level
- scenario context
- stem pattern
- distractor strategy

Avoid generic examples such as `What is 2 + 2?`.

### `factual_restraint.py`

Generates questions where the desired answer avoids overclaiming. The prompt should vary:

- uncertainty type
- question domain
- safe-answer style
- unsupported premise pattern

---

## Editing Guidelines

When changing prompts:

1. Keep the `{"items": [...]}` contract.
2. Keep fields compatible with `slm_synth/schemas.py`.
3. Run a signal-level smoke test.
4. Report duplicates before and after validation.
5. Avoid increasing batch size as a first fix for throughput.

Useful smoke test:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate SIGNAL=educational_qa_mcq
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
```
