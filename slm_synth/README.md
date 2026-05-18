# `slm_synth`

Core package for the SLM synthetic-data pipeline.

This package implements:

```text
generate -> validate -> dedup -> push_hf
```

It also includes shared path resolution, LLM calls, retry/backoff logic, JSON repair, diversity controls, duplicate reporting, and schema validation.

---

## Important Modules

```text
slm_synth/
├── paths.py              # Shared config loading and output-dir resolution.
├── schemas.py            # JSON schema validation for each signal.
├── repair.py             # Lightweight record repair/normalization.
├── diversity.py          # Per-batch diversity context generation.
├── llm.py                # Groq/OpenAI-compatible LLM client wrapper.
├── rate_limit.py         # Request pacing and jitter helpers.
├── writer.py             # JSONL writer helpers.
├── generate.py           # Batched generation pipeline.
├── validate.py           # Schema validation stage.
├── dedup.py              # Exact/fuzzy dedup stage; exact-only by default.
├── report_duplicates.py  # Exact duplicate reporting by stage.
├── push_hf.py            # HF upload + dataset card generation.
└── sources/              # Signal-specific prompt builders.
```

---

## Path Resolution

All config-based stages should use:

```python
from slm_synth.paths import load_yaml_config, resolve_output_dir
```

`resolve_output_dir` handles config values such as:

```yaml
output_dir: "${DATA_DIR}/<run_name>"
```

If `DATA_DIR` is not set, it defaults to:

```text
data/runs/<run_name>
```

This keeps `bootstrap_dirs.py`, `generate.py`, `validate.py`, `dedup.py`, and `push_hf.py` aligned.

---

## Generation

`generate.py` supports:

- config-based execution
- optional single-signal execution
- JSON object batches
- Groq JSON mode
- per-signal batch sizes
- concurrency
- request backoff and jitter
- recursive batch splitting on parse/schema failure
- rejected-batch quarantine
- diversity context injection

Run all signals:

```bash
python -m slm_synth.generate --config configs/synthetic.yaml
```

Run one signal:

```bash
python -m slm_synth.generate --config configs/synthetic.yaml --signal task_code
```

---

## Validation

`validate.py` reads from:

```text
<output_dir>/raw/*.jsonl
```

and writes to:

```text
<output_dir>/validated/*.jsonl
```

Run:

```bash
python -m slm_synth.validate --config configs/synthetic.yaml
```

---

## Deduplication

`dedup.py` reads from:

```text
<output_dir>/validated/*.jsonl
```

and writes to:

```text
<output_dir>/deduped/*.jsonl
```

Synthetic data should use exact-only deduplication:

```yaml
dedup:
  mode: "exact"
  enable_exact: true
  enable_fuzzy: false
```

Fuzzy dedup is intentionally not the default because it can collapse template-like but useful synthetic variation.

Run:

```bash
python -m slm_synth.dedup --config configs/synthetic.yaml
```

---

## Duplicate Reporting

Use `report_duplicates.py` before and after dedup:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage validated
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

This is a key scaling check. High exact duplicate rates mean prompt diversity is failing and the run should not be scaled.

---

## Hugging Face Push

`push_hf.py` reads deduped files, generates a dataset card, and uploads the dataset to Hugging Face.

It loads tokens from `.env`:

```text
HF_TOKEN=...
```

or:

```text
HUGGINGFACE_HUB_TOKEN=...
```

Run:

```bash
python -m slm_synth.push_hf --config configs/synthetic.yaml
```

Skip dataset card upload only for debugging:

```bash
python -m slm_synth.push_hf --config configs/synthetic.yaml --skip-card
```

---

## Supported Models

Known-good models:

```text
llama-3.1-8b-instant
llama-3.3-70b-versatile
```

Do not assume other Groq models are production-ready for this pipeline. Structured generation quality varies by model.
