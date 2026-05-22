# `slm_synth` Package

This package implements the synthetic data pipeline.

## Main modules

| Module | Purpose |
|---|---|
| `generate.py` | Runs two-pass candidate generation and independent response completion for one or more signals. |
| `llm.py` | Groq/OpenAI-compatible chat backend wrapper with JSON object mode and retry handling for both passes. |
| `rate_limit.py` | Request pacing and jitter helpers. |
| `diversity.py` | Per-batch diversity context generation. |
| `validate.py` | Schema and signal-specific validation from `raw/` to `validated/`; math MCQs receive numeric verification. |
| `dedup.py` | Exact deduplication from `validated/` to `deduped/`. |
| `report_duplicates.py` | Duplicate and bad-JSON reporting for pipeline stages. |
| `push_hf.py` | Hugging Face upload and dataset-card generation. |
| `paths.py` | Shared config loading and output-path resolution. |
| `model_support.py` | Lightweight supported-model warning helper. |
| `schemas.py` | Signal schema validators. |
| `repair.py` | Lightweight final-record normalization helpers; answer keys are supplied by the response pass, not invented during repair. |
| `writer.py` | JSONL writer utility. |

## Two-pass generation

For every signal, the candidate model first writes an unanswered task or question. A separate response-model call then answers the fixed candidate and produces the final record written to `raw/`. The default template configures both roles as `llama-3.1-8b-instant`; `candidate_model` and `response_model` can be overridden per signal.

## Pipeline stages

```bash
python -m slm_synth.generate --config configs/synthetic.yaml
python -m slm_synth.validate --config configs/synthetic.yaml
python -m slm_synth.dedup --config configs/synthetic.yaml
python -m slm_synth.push_hf --config configs/synthetic.yaml
```

Each stage reads the same config and uses the shared output-directory resolver.

## Model support

The package is validated with:

- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`

If another model is configured, generation prints a warning but continues. This is intentional: the supported path is explicit, while experiments remain possible.

## Dedup behavior

The package generates five signals, including separate `educational_qa_mcq_math` and `educational_qa_mcq_general` outputs. Math MCQs use temporary raw-stage verification fields that are removed during validation; general MCQs remain non-math and context-grounded.

Synthetic data uses exact deduplication by default. Fuzzy deduplication is not recommended for these signals because it can collapse useful template-like variation.
