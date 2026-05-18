# `slm_synth` Package

This package implements the synthetic data pipeline.

## Main modules

| Module | Purpose |
|---|---|
| `generate.py` | Runs synthetic generation for one or more signals. |
| `llm.py` | Groq/OpenAI-compatible chat backend wrapper with JSON object mode and retry handling. |
| `rate_limit.py` | Request pacing and jitter helpers. |
| `diversity.py` | Per-batch diversity context generation. |
| `validate.py` | Schema validation from `raw/` to `validated/`. |
| `dedup.py` | Exact deduplication from `validated/` to `deduped/`. |
| `report_duplicates.py` | Duplicate and bad-JSON reporting for pipeline stages. |
| `push_hf.py` | Hugging Face upload and dataset-card generation. |
| `paths.py` | Shared config loading and output-path resolution. |
| `model_support.py` | Lightweight supported-model warning helper. |
| `schemas.py` | Signal schema validators. |
| `repair.py` | Lightweight record repair/normalization helpers. |
| `writer.py` | JSONL writer utility. |

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

Synthetic data uses exact deduplication by default. Fuzzy deduplication is not recommended for these signals because it can collapse useful template-like variation.
