# Prompts

Prompt modules define the two-pass candidate and response tasks and schema guidance for each synthetic signal.

## Contract

Both the candidate pass and the independent response pass ask the model to return a JSON object with an `items` array:

```json
{
  "items": [
    { "...": "..." }
  ]
}
```

The pipeline intentionally avoids bare top-level JSON arrays because JSON object output is more reliable with the supported Groq models.

## Signals

| Signal | File | Purpose |
|---|---|---|
| Signal | Candidate pass | Response pass |
|---|---|---|
| `arithmetic` | Unanswered integer question | Steps and exact answer |
| `task_code` | Task specification only | Plan and Python function |
| `educational_qa_mcq_math` | Question and choices without a key | Corrected final choices/key/explanation plus numeric verification metadata |
| `educational_qa_mcq_general` | Grounded question and choices without a key | Corrected final choices/key/explanation |
| `factual_restraint` | Question only | Safe restrained answer |

## Diversity

Per-batch diversity context is applied to the candidate-authoring pass. The independent response pass answers the resulting fixed candidates. Mathematical MCQ responses carry temporary raw-stage verification fields that are checked and removed before export, and explanations containing answer-key or generation-error commentary are rejected. General MCQ responses must ground the answer in explicit supplied evidence.

## Prompt rules

Prompts should:

- keep the top-level JSON object contract,
- avoid markdown and code fences,
- keep string fields JSON-safe,
- produce concise records,
- avoid generic repeated examples,
- preserve the schema expected by `slm_synth.schemas`.

When changing prompts, run a small generation test and duplicate report before scaling:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```
