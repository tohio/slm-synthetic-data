# Prompts

Prompt modules define the generation task and schema guidance for each synthetic signal.

## Contract

The generator asks the model to return a JSON object with an `items` array:

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
| `arithmetic` | `arithmetic.py` | Integer arithmetic questions with short reasoning steps. |
| `task_code` | `task_code.py` | Small Python programming tasks with plans and code snippets. |
| `educational_qa_mcq_math` | `educational_qa_mcq_math.py` | Machine-verifiable mathematical multiple-choice questions. |
| `educational_qa_mcq_general` | `educational_qa_mcq_general.py` | Non-math educational multiple-choice questions grounded in supplied context. |
| `factual_restraint` | `factual_restraint.py` | Questions that require cautious answers instead of unsupported claims. |

## Diversity

Per-batch diversity context is added by `slm_synth.diversity`. It rotates attributes such as topic, difficulty, format, scenario, and answer style. Mathematical MCQs also carry temporary raw-stage verification fields that are checked and removed before export, and explanations containing answer-key or generation-error commentary are rejected. General MCQs do not use numeric verification fields; their prompts require the answer to be chosen from explicit supplied evidence without a forced answer position.

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
