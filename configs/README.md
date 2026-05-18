# Configuration

This directory contains configuration helpers and templates for the SLM synthetic-data pipeline.

The generated runtime config is:

```text
configs/synthetic.yaml
```

`configs/synthetic.yaml` is produced from:

```text
configs/synthetic_template.yaml
```

using:

```text
configs/configure_synthetic.py
```

---

## Recommended Configuration Flow

Generate a config with:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

For a full 5M-token validation run:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Then bootstrap the run directory:

```bash
python bootstrap_dirs.py
```

---

## Profiles

| Profile | Model | Use Case |
|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Faster bulk generation. |
| `balanced` | `llama-3.1-8b-instant` | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Higher-quality smaller runs or audit generation. |

The codebase is validated for two Groq-hosted Llama models:

```text
llama-3.1-8b-instant
llama-3.3-70b-versatile
```

Other Groq models may work experimentally, but they are not treated as production-supported. Models that do not reliably produce JSON object batches are not suitable for large synthetic runs without additional prompt/parser work.

---

## Important Generated Fields

`configure_synthetic.py` updates:

```text
run_name
output_dir
target_total_tokens
backend.model
backend.max_tokens
backend.temperature
backend.top_p
backend.parallel_requests
backend.service_tier
mix.<signal>.batch_size
mix.<signal>.parallel_requests
mix.<signal>.max_tokens
rate_limit.max_concurrency
generation.batch_size
generation.avg_tokens_per_sample
```

The active run writes to:

```text
${DATA_DIR}/<run_name>
```

If `DATA_DIR` is not exported, the shared path resolver uses:

```text
data/runs/<run_name>
```

---

## Backend Settings

The backend section controls Groq generation:

```yaml
backend:
  provider: "groq"
  model: "llama-3.1-8b-instant"
  max_tokens: 1536
  temperature: 0.45
  top_p: 0.95
  parallel_requests: 8
  json_mode: true
  service_tier: "flex"
```

`json_mode: true` is important. The prompt wrapper asks for:

```json
{"items": [...]}
```

rather than a top-level JSON array.

---

## Retry and Backoff Settings

The generated config includes request-level retries:

```yaml
backend:
  request_timeout_seconds: 120
  retries:
    max_request_retries: 12
    retry_backoff_initial_seconds: 1.0
    retry_backoff_max_seconds: 45.0
    retry_backoff_multiplier: 2.0
    retry_jitter_ratio: 0.35
```

This is used to handle transient capacity, timeout, rate-limit, and server errors from Groq, especially when using `SERVICE_TIER=flex`.

---

## Signal Mix

The default mix is:

```yaml
mix:
  arithmetic:
    share: 0.30
  task_code:
    share: 0.30
  educational_qa_mcq:
    share: 0.30
  factual_restraint:
    share: 0.10
```

Each signal has its own model, batch size, max token budget, and average tokens-per-sample estimate. This allows `task_code` to stay more conservative while shorter signals can use larger batches.

---

## Deduplication Settings

Synthetic data should use exact deduplication by default:

```yaml
dedup:
  mode: "exact"
  enable_exact: true
  enable_fuzzy: false
  fuzzy_enabled: false
```

Fuzzy MinHash deduplication is intentionally disabled. It can remove useful synthetic variation because generated records often share schemas, phrasing, and reasoning structure.

---

## Hugging Face Export Settings

The export section controls HF push:

```yaml
export:
  push_to_hf: true
  hf_repo: "tohio/slm-synthetic"
  private: false
  include_manifests: true
```

`push_hf.py` loads credentials from `.env`:

```text
HF_TOKEN=...
```

or:

```text
HUGGINGFACE_HUB_TOKEN=...
```

The push command uploads deduped JSONL files and a generated dataset card.

---

## Useful Commands

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
make push
```
