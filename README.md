# SLM Synthetic Data

Synthetic data generation pipeline for the SLM project.

This repository generates structured synthetic examples for pretraining and post-training data experiments. It currently supports four signal families:

- `arithmetic` — integer arithmetic, word problems, comparisons, missing-value problems, and compact reasoning steps.
- `task_code` — beginner/intermediate Python tasks with short plans and code snippets.
- `educational_qa_mcq` — scenario-based multiple-choice questions with explanations.
- `factual_restraint` — questions that reward cautious answers and discourage unsupported claims.

The current pipeline is optimized for Groq-hosted Llama models, JSON-object batched generation, exact deduplication, and Hugging Face dataset publishing.

![Architecture](docs/architecture.png)

---

## Current Status

The pipeline has been validated end to end with:

```text
generate -> validate -> dedup -> push_hf
```

Known-good generation settings:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
make push
```

The current implementation includes:

- JSON object generation contract: `{"items": [...]}`.
- Groq JSON object mode support.
- Groq Flex service-tier support.
- Exponential backoff with jitter for rate-limit and Flex capacity errors.
- Per-batch diversity controls to avoid repeated examples.
- Exact-only deduplication by default.
- Duplicate-rate reporting by stage.
- Config-based `generate`, `validate`, `dedup`, and `push_hf` commands.
- Shared path resolution for `${DATA_DIR}/<run_name>`.
- Hugging Face push with `.env` token loading.
- Hugging Face dataset card generation and upload.

---

## Supported Models

The repo is intentionally **not** model-agnostic for production generation.

The Groq backend is configurable, but large-scale synthetic generation depends on models reliably following a strict JSON/batching contract. The following models are the only models currently considered known-good:

| Model | Intended Use |
|---|---|
| `llama-3.1-8b-instant` | Recommended default for bulk generation. |
| `llama-3.3-70b-versatile` | Higher-quality generation, smaller runs, audits, or targeted repair. |

Models such as `openai/gpt-oss-20b` are not currently supported for production generation in this repo because they have been observed to break the JSON-array/object contract more often. They may be used experimentally, but do not assume they will scale without prompt/parser changes.

Recommended default:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

---

## Repository Structure

```text
slm-synthetic-data/
├── README.md
├── Makefile
├── requirements.txt
├── pytest.ini
├── bootstrap_dirs.py
│
├── configs/
│   ├── README.md
│   ├── configure_synthetic.py
│   └── synthetic_template.yaml
│
├── docs/
│   ├── README.md
│   └── architecture.png
│
├── prompts/
│   ├── README.md
│   ├── wrapper.py
│   ├── arithmetic.py
│   ├── task_code.py
│   ├── educational_qa_mcq.py
│   └── factual_restraint.py
│
├── slm_synth/
│   ├── README.md
│   ├── paths.py
│   ├── schemas.py
│   ├── repair.py
│   ├── diversity.py
│   ├── llm.py
│   ├── rate_limit.py
│   ├── writer.py
│   ├── generate.py
│   ├── validate.py
│   ├── dedup.py
│   ├── report_duplicates.py
│   ├── push_hf.py
│   └── sources/
│       ├── arithmetic.py
│       ├── task_code.py
│       ├── educational_qa_mcq.py
│       └── factual_restraint.py
│
├── tests/
│   ├── README.md
│   ├── test_generate.py
│   ├── test_validate.py
│   ├── test_dedup.py
│   └── test_schemas.py
│
└── data/
    └── runs/
        └── <run_name>/
            ├── raw/
            ├── validated/
            ├── deduped/
            ├── rejected/
            └── manifests/
```

`data/` is generated output and should not be committed.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.sample .env
```

At minimum, set:

```text
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

`HUGGINGFACE_HUB_TOKEN` is also accepted for Hugging Face pushes.

Never commit `.env`.

---

## Configuration

Generate `configs/synthetic.yaml` from the template:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Common profiles:

| Profile | Default Model | Purpose |
|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Faster bulk generation. |
| `balanced` | `llama-3.1-8b-instant` | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Higher quality, slower/more expensive. |

Useful overrides:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
make configure PROFILE=quality TOKENS=1000000 BATCH=4 CONCURRENCY=4 SERVICE_TIER=flex
make configure MODEL=llama-3.3-70b-versatile TOKENS=1000000 BATCH=4 CONCURRENCY=4
```

The generated config writes outputs to:

```text
${DATA_DIR}/<run_name>
```

When `DATA_DIR` is not exported, the shared resolver defaults to:

```text
data/runs/<run_name>
```

---

## Pipeline

### 1. Bootstrap directories

```bash
python bootstrap_dirs.py
```

Creates:

```text
data/runs/<run_name>/raw
data/runs/<run_name>/validated
data/runs/<run_name>/deduped
data/runs/<run_name>/rejected
data/runs/<run_name>/manifests
```

### 2. Generate

Generate all signals:

```bash
make generate
```

Generate one signal:

```bash
make generate SIGNAL=educational_qa_mcq
```

The generator writes JSONL files to:

```text
data/runs/<run_name>/raw/*.jsonl
```

### 3. Report duplicates

Before validation or dedup, inspect exact duplicate rates:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```

Healthy synthetic generation should usually stay below roughly 5% exact duplicates. If the duplicate rate is much higher, fix generation diversity before scaling.

### 4. Validate

```bash
make validate
```

Validated records are written to:

```text
data/runs/<run_name>/validated/*.jsonl
```

Rejected records remain under:

```text
data/runs/<run_name>/rejected/*.jsonl
```

### 5. Deduplicate

```bash
make dedup
```

Synthetic data uses exact deduplication by default. Fuzzy MinHash deduplication is intentionally disabled because synthetic records often share templates and schemas by design.

Deduped records are written to:

```text
data/runs/<run_name>/deduped/*.jsonl
```

### 6. Push to Hugging Face

```bash
make push
```

`push_hf.py` loads Hugging Face credentials from `.env`, uploads the deduped JSONL files, and uploads a generated dataset card as `README.md`.

The repo target is configured in `configs/synthetic.yaml`:

```yaml
export:
  push_to_hf: true
  hf_repo: "tohio/slm-synthetic"
  private: false
```

---

## JSON Output Contract

The model is instructed to return a JSON object, not a bare JSON array:

```json
{
  "items": [
    { "type": "..." }
  ]
}
```

The parser accepts this object and extracts the `items` array. This format is more reliable with Groq JSON object mode than requesting a top-level array.

---

## Deduplication Policy

Use exact deduplication by default for all synthetic signals:

```yaml
dedup:
  mode: "exact"
  enable_exact: true
  enable_fuzzy: false
  fuzzy_enabled: false
```

Do not use fuzzy MinHash deduplication for synthetic data unless you are running a specific experiment. Fuzzy dedup can remove useful synthetic variation because many examples intentionally share structure.

Recommended interpretation:

| Exact Duplicate Rate | Meaning |
|---:|---|
| `0-5%` | Healthy. |
| `5-15%` | Inspect signal prompts. |
| `15%+` | Fix diversity before scaling. |
| `50%+` | Do not scale; generation is collapsing. |

---

## Scaling Guidance

Recommended progression:

```text
200K tokens  -> smoke test
5M tokens    -> full pipeline test
50M tokens   -> long-run stability test
600M tokens  -> production-scale generation
```

Recommended starting settings:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

If Groq Flex capacity errors appear, keep `BATCH=4` and reduce concurrency before changing the prompt contract:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=4 SERVICE_TIER=flex
```

The code includes exponential backoff with jitter for transient rate-limit, timeout, server, and Flex-capacity errors.

---

## Testing

Install test dependencies if needed:

```bash
pip install pytest
```

Run:

```bash
make test
```

Or:

```bash
python -m pytest -q
```

---

## Current Commit Messages

Useful commit messages for the recent changes:

```text
Add scalable Groq JSON-object generation path
Add exponential backoff for Groq Flex capacity errors
Fix shared pipeline path resolution
Default synthetic dedup to exact matching
Load HF credentials from dotenv during push
Add per-batch diversity controls for synthetic generation
Improve educational MCQ generation diversity
Upload dataset card with Hugging Face push
Refresh documentation for current synthetic pipeline
```
