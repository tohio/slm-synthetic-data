# SLM Synthetic Data

This repository generates small, structured synthetic datasets for SLM/LLM training and evaluation experiments.

The pipeline produces JSONL records for four signals:

- `arithmetic`
- `task_code`
- `educational_qa_mcq`
- `factual_restraint`

The current implementation is optimized for Groq-hosted Llama models, JSON object output, deterministic validation, exact deduplication, and Hugging Face dataset publishing.

## Supported models

This project is validated with the following Groq models:

| Model | Use |
|---|---|
| `llama-3.1-8b-instant` | Recommended default for scalable bulk generation. |
| `llama-3.3-70b-versatile` | Higher-quality option for smaller or quality-focused runs. |

Other models may work, but they are not currently validated. The pipeline depends on reliable JSON object output and strict schema following. Models that do not consistently satisfy that contract are not recommended for production generation.

If an unvalidated model is configured, the code prints a warning but does not hard-fail. This keeps experimentation possible while making the supported path explicit.

## Pipeline

```text
configure -> bootstrap -> generate -> report duplicates -> validate -> dedup -> push
```

| Stage | Command | Output |
|---|---|---|
| Configure | `make configure ...` | `configs/synthetic.yaml` |
| Bootstrap | `python bootstrap_dirs.py` | `data/runs/<run_id>/...` directories |
| Generate | `make generate` | `raw/*.jsonl` |
| Duplicate report | `python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw` | duplicate/bad JSON report |
| Validate | `make validate` | `validated/*.jsonl` |
| Dedup | `make dedup` | `deduped/*.jsonl` |
| Push | `make push` | Hugging Face dataset files and dataset card |

## Quickstart

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
```

Configure a small run:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
```

Check duplicate rate before continuing:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```

Validate and deduplicate:

```bash
make validate
make dedup
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

Push to Hugging Face:

```bash
make push
```

## Profiles

Profiles choose a default model and runtime posture. `speed` and `balanced` both use `llama-3.1-8b-instant`; they differ by throughput settings, not model family.

| Profile | Default model | Typical posture | Purpose |
|---|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Higher concurrency, throughput-oriented settings | Fastest bulk generation when retries are acceptable. |
| `balanced` | `llama-3.1-8b-instant` | Moderate concurrency, JSON object mode, diversity controls, backoff | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Lower concurrency, higher-quality model | Smaller quality-focused runs or comparison runs. |

Recommended production-style starting point:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Scale gradually:

```text
200K smoke test -> 5M pipeline test -> 50M stability test -> larger runs
```

## Data layout

Runs are written under `data/runs/<run_id>/`:

```text
data/runs/<run_id>/
├── raw/          # generated JSONL records
├── validated/    # schema-valid JSONL records
├── deduped/      # exact-deduped JSONL records, preferred training input
├── rejected/     # rejected batches or invalid records
└── manifests/    # generated metadata, including Hugging Face README.md
```

The configured output directory uses `${DATA_DIR}` by default. If `DATA_DIR` is not exported, the resolver defaults to `data/runs`.

## Dedup policy

Synthetic data should use exact deduplication by default.

Fuzzy MinHash deduplication can be destructive for synthetic datasets because records often share schema, wording style, code skeletons, reasoning formats, and safe-answer templates by design.

Recommended downstream input:

```text
data/runs/<run_id>/deduped/*.jsonl
```

Use duplicate reports to catch generation quality issues before scaling:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```

General interpretation:

| Exact duplicate rate | Interpretation |
|---:|---|
| `0-5%` | Healthy. |
| `5-15%` | Inspect prompts and diversity controls. |
| `15%+` | Fix generation diversity before scaling. |

## Hugging Face publishing

`make push` uploads the deduped JSONL files and a generated dataset card:

```text
README.md
arithmetic.jsonl
task_code.jsonl
educational_qa_mcq.jsonl
factual_restraint.jsonl
```

The push step reads credentials from `.env` or the shell:

```bash
HF_TOKEN=...
# or
HUGGINGFACE_HUB_TOKEN=...
```

Set the repository target when configuring:

```bash
make configure PROFILE=balanced TOKENS=5000000 HF_REPO=tohio/slm-synthetic
```

## Repository structure

```text
.
├── README.md
├── Makefile
├── bootstrap_dirs.py
├── configs/
│   ├── README.md
│   ├── configure_synthetic.py
│   └── synthetic_template.yaml
├── docs/
│   ├── README.md
│   └── TODO.md
├── prompts/
│   ├── README.md
│   ├── arithmetic.py
│   ├── educational_qa_mcq.py
│   ├── factual_restraint.py
│   ├── task_code.py
│   └── wrapper.py
├── slm_synth/
│   ├── README.md
│   ├── dedup.py
│   ├── diversity.py
│   ├── generate.py
│   ├── llm.py
│   ├── model_support.py
│   ├── paths.py
│   ├── push_hf.py
│   ├── rate_limit.py
│   ├── repair.py
│   ├── report_duplicates.py
│   ├── schemas.py
│   ├── validate.py
│   └── writer.py
└── tests/
    ├── README.md
    ├── test_dedup.py
    ├── test_generate.py
    ├── test_schemas.py
    └── test_validate.py
```

## Notes

- Generated data should not be committed to the repository.
- Keep `.env` out of version control.
- Use exact deduped records for downstream training unless a specific experiment requires otherwise.
- Validate small runs end-to-end before increasing token targets.
