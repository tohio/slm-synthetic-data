# slm-synthetic-data

Synthetic dataset generation pipeline for small language model training. The project generates structured JSONL data for arithmetic reasoning, beginner task-code examples, educational multiple-choice questions, and factual-restraint responses.

The pipeline is designed for reproducible dataset runs:

```text
configure -> bootstrap -> generate -> duplicate report -> validate -> exact dedup -> push to Hugging Face
```

> Status: The pipeline is operational with Groq Llama models. The validated production path uses JSON object generation, per-batch diversity controls, Groq Flex backoff, exact-only deduplication, and Hugging Face dataset-card upload.

---

## Overview

Most data pipelines treat synthetic generation as a one-off script. This repository treats it as a repeatable data production workflow.

Each run produces a self-contained directory under `data/runs/<run_id>/` with raw, validated, deduped, rejected, and manifest outputs. Configuration is generated from a template so the same commands can be used for small smoke tests, medium validation runs, and larger production-style runs.

The core goals are:

- Generate schema-valid synthetic records at scale.
- Keep generation stable under Groq Flex capacity pressure.
- Preserve useful synthetic variation by using exact-only deduplication.
- Track duplicate rates before and after deduplication.
- Push clean JSONL datasets and a dataset card to Hugging Face.

---

## Key Decisions

- **Use validated Groq Llama models by default.** The pipeline is validated with `llama-3.1-8b-instant` for scalable bulk generation and `llama-3.3-70b-versatile` for smaller quality-focused runs. Other models may work, but they are not currently validated for production-scale generation.
- **Use JSON object batches.** Model responses are expected to be JSON objects with an `items` array, not bare JSON arrays. This keeps the output contract compatible with Groq JSON object mode and local schema validation.
- **Scale with moderate batches plus concurrency.** The recommended production posture is `BATCH=4` with controlled concurrency and backoff, rather than very large batches.
- **Use Groq Flex with backoff for larger runs.** Flex capacity errors are treated as retryable conditions with exponential backoff and jitter.
- **Deduplicate synthetic data with exact matching by default.** Fuzzy MinHash dedup can collapse useful template-like synthetic variation and should remain disabled unless explicitly running an experiment.
- **Track duplicate rate before scaling.** Raw and validated duplicate reports are part of the scaling workflow. High exact-duplicate rates indicate a prompt/diversity issue and should be fixed before larger runs.
- **Use `deduped/` as the downstream training input.** `raw/` and `validated/` are useful for inspection, but the exact-deduped JSONL files are the preferred exported dataset.
- **Publish with a dataset card.** Hugging Face pushes include the generated JSONL files and a dataset card derived from the active config and output files.

## Choosing a run size

All sizes use the same pipeline. The difference is the configured token target.

| Size | Command token target | Purpose |
|---|---:|---|
| Smoke | `200000` | Fast validation of config, prompts, JSON parsing, validation, and dedup. |
| Small | `5000000` | End-to-end quality check before scaling. |
| Medium | `50000000` | Longer run for throughput, duplicate-rate, and retention validation. |
| Large | `600000000` | Production-scale target after medium-run metrics are acceptable. |

Recommended progression:

```bash
200K -> 5M -> 50M -> larger production run
```

Do not jump directly to a large run before checking rejection rate, duplicate rate, validation retention, and dedup retention.

---

## Supported models

This project is validated with the following Groq models:

| Model | Use |
|---|---|
| `llama-3.1-8b-instant` | Recommended default for scalable bulk generation. |
| `llama-3.3-70b-versatile` | Higher-quality option for smaller or quality-focused runs. |

Other models may work, but they are not currently validated. The pipeline depends on reliable JSON object output, strict schema following, correct batch counts, and valid escaped string values.

The code emits a non-blocking warning when a model outside the validated list is selected.

---

## Generation profiles

Profiles select a default model and runtime posture. `speed` and `balanced` both use the same default model; they differ by throughput settings.

| Profile | Default model | Runtime posture | Purpose |
|---|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Higher concurrency, throughput-oriented settings | Fast bulk generation when occasional retries are acceptable. |
| `balanced` | `llama-3.1-8b-instant` | Moderate concurrency, diversity enabled, backoff enabled | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Lower concurrency, higher-quality model | Smaller quality-focused or comparison runs. |

---

## Signals

The default mix generates four signal families:

| Signal | Description | Output file |
|---|---|---|
| `arithmetic` | Integer arithmetic reasoning records. | `arithmetic.jsonl` |
| `task_code` | Beginner programming tasks with short Python solutions. | `task_code.jsonl` |
| `educational_qa_mcq` | Educational multiple-choice questions with explanations. | `educational_qa_mcq.jsonl` |
| `factual_restraint` | Questions that require cautious, non-hallucinated answers. | `factual_restraint.jsonl` |

Each signal has its own prompt, schema, batch size, token budget, and diversity controls.

---

## Tech stack

| Stage | Tooling |
|---|---|
| Config generation | Python + YAML templates |
| LLM backend | Groq API |
| Environment loading | `python-dotenv` |
| Generation format | JSON object batches with an `items` array |
| Validation | Local JSON/schema validation |
| Deduplication | Exact matching by default |
| Duplicate reporting | Local JSONL duplicate scanner |
| Dataset publishing | `huggingface_hub` |
| Workflow entrypoints | `make` targets |

---

## Repo structure

```text
slm-synthetic-data/
├── bootstrap_dirs.py              # creates run directory structure from configs/synthetic.yaml
├── configs/
│   ├── configure_synthetic.py     # generates configs/synthetic.yaml from profile/CLI args
│   ├── synthetic_template.yaml    # source template for generated configs
│   ├── synthetic.yaml             # generated active run config
│   └── README.md
├── docs/
│   ├── COMMANDS.md                # command reference
│   ├── README.md                  # documentation index
│   └── TODO.md                    # public project backlog
├── prompts/
│   ├── arithmetic.py
│   ├── educational_qa_mcq.py
│   ├── factual_restraint.py
│   ├── task_code.py
│   ├── wrapper.py
│   └── README.md
├── slm_synth/
│   ├── diversity.py               # per-batch diversity controls
│   ├── dedup.py                   # exact/fuzzy dedup entrypoint
│   ├── generate.py                # generation runner
│   ├── llm.py                     # Groq client wrapper
│   ├── model_support.py           # validated-model warning helper
│   ├── paths.py                   # shared config/path resolution
│   ├── push_hf.py                 # Hugging Face upload + dataset card
│   ├── rate_limit.py              # pacing/backoff helpers
│   ├── repair.py                  # record repair helpers
│   ├── report_duplicates.py       # duplicate reporting CLI
│   ├── schemas.py                 # signal schemas
│   ├── validate.py                # validation runner
│   ├── writer.py                  # JSONL writing helpers
│   ├── sources/
│   │   ├── arithmetic.py
│   │   ├── educational_qa_mcq.py
│   │   ├── factual_restraint.py
│   │   └── task_code.py
│   └── README.md
├── tests/
│   └── README.md
├── Makefile
├── requirements.txt
└── .env.sample
```

Generated run data is written under:

```text
data/runs/<run_id>/
├── raw/
├── validated/
├── deduped/
├── rejected/
└── manifests/
```

---

## Getting started

### Prerequisites

- Python 3.12+
- Groq account and API key
- Hugging Face account and token if using `make push`
- `make`

### Installation

```bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.sample .env
vi .env
```

Required for generation:

```env
GROQ_API_KEY=...
```

Required for Hugging Face push:

```env
HF_TOKEN=...
# or
HUGGINGFACE_HUB_TOKEN=...
```

Optional:

```env
DATA_DIR=data/runs
```

If `DATA_DIR` is not set, the shared path resolver uses `data/runs`.

---

## Quickstart

Run a small smoke test:

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

Run a larger validation run:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
```

Push the deduped dataset to Hugging Face:

```bash
make push
```

`make push` uploads:

```text
README.md                 # generated dataset card
arithmetic.jsonl
task_code.jsonl
educational_qa_mcq.jsonl
factual_restraint.jsonl
```

---

## Common commands

See the full command reference:

```text
docs/COMMANDS.md
```

Common targets:

| Command | Purpose |
|---|---|
| `make configure` | Generate `configs/synthetic.yaml`. |
| `python bootstrap_dirs.py` | Create run directories. |
| `make generate` | Generate all configured signals. |
| `make generate SIGNAL=arithmetic` | Generate one signal only. |
| `make validate` | Validate raw records into `validated/`. |
| `make dedup` | Deduplicate validated records into `deduped/`. |
| `make push` | Push deduped JSONL files and dataset card to Hugging Face. |

---

## Resume after interruption

Do not delete `data` if you want to continue an interrupted run.

Activate the environment:

```bash
cd ~/slm-synthetic-data
source .venv/bin/activate || source venv/bin/activate
export PYTHONUNBUFFERED=1
```

Inspect current files:

```bash
grep -n "run_name\|output_dir" configs/synthetic.yaml

for f in data/runs/*/raw/*.jsonl; do
  echo "$f $(wc -l < "$f")"
done
```

Continue with a specific signal:

```bash
make generate SIGNAL=educational_qa_mcq
```

If a signal needs to be rerun cleanly, remove only that signal file:

```bash
rm -f data/runs/<run_id>/raw/arithmetic.jsonl
rm -f data/runs/<run_id>/rejected/arithmetic.jsonl
make generate SIGNAL=arithmetic
```

---

## Deduplication policy

Synthetic data intentionally shares structure. Fuzzy deduplication can remove useful variation.

Default policy:

```text
Use exact deduplication for synthetic data.
Do not enable fuzzy MinHash deduplication unless running an explicit experiment.
```

Recommended thresholds before scaling:

| Metric | Target |
|---|---:|
| Bad JSON | `0` |
| Rejected batches | near `0` |
| Overall exact duplicate rate | under `5%` |
| Validation rejection rate | under `1%` |
| Dedup retention | above `95%` |

Check duplicate rates:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage validated
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

---

## Dataset format

Each JSONL file contains one JSON object per line. All records include a `type` field.

Example arithmetic record:

```json
{
  "type": "arithmetic",
  "question": "A box has 18 red marbles and 27 blue marbles. How many marbles are there in total?",
  "steps": ["Add the red and blue marbles.", "18 + 27 = 45"],
  "answer": "45"
}
```

Example MCQ record:

```json
{
  "type": "educational_qa_mcq",
  "question": "A student measures the temperature of water before and after heating it. Which variable is being observed?",
  "choices": ["The color of the cup", "The water temperature", "The room size", "The clock time"],
  "correct_index": 1,
  "explanation": "The observed variable is the water temperature because it is measured before and after heating."
}
```

---

## Hugging Face publishing

Configure the repository target:

```bash
make configure PROFILE=balanced TOKENS=5000000 HF_REPO=tohio/slm-synthetic
```

Then push:

```bash
make push
```

The push stage loads credentials from `.env`, writes a dataset card to:

```text
data/runs/<run_id>/manifests/README.md
```

and uploads the card as the repository `README.md`.

---

## Development checks

Compile key files:

```bash
python -m py_compile   bootstrap_dirs.py   configs/configure_synthetic.py   slm_synth/*.py   slm_synth/sources/*.py   prompts/*.py
```

Run tests:

```bash
python -m pytest -q
```

See:

```text
tests/README.md
```

---

## Troubleshooting

### `make: python: No such file or directory`

Activate the virtual environment:

```bash
source .venv/bin/activate || source venv/bin/activate
which python
```

### Generation appears to hang when piped through `tee`

Use unbuffered Python:

```bash
export PYTHONUNBUFFERED=1
make generate
```

### Groq Flex capacity errors

The generator includes exponential backoff and jitter for temporary capacity errors. If rejected batches climb, reduce concurrency:

```bash
make configure PROFILE=balanced TOKENS=50000000 BATCH=4 CONCURRENCY=6 SERVICE_TIER=flex
```

### Unsupported model warning

Only two Groq models are currently validated. The warning is non-blocking, but production runs should use a validated model.

---

## Documentation

- [Command reference](docs/COMMANDS.md) — common `make` targets, pipeline commands, resume workflows, and troubleshooting commands.
- [Disk setup](docs/DISK_SETUP.md) — storage layout and disk preparation for larger synthetic-data runs.
- [Configuration guide](configs/README.md) — profiles, model settings, output paths, and Hugging Face settings.
- [Prompt guide](prompts/README.md) — signal prompts, JSON output contract, and diversity controls.
- [Package guide](slm_synth/README.md) — internal module layout and developer notes.
- [Test guide](tests/README.md) — offline checks, live smoke tests, and validation expectations.
- [Project docs](docs/README.md) — documentation index.

## License

TBD.
