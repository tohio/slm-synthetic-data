# SLM Synthetic Data

This repository generates synthetic data signals that can be mixed into the training corpus for small language model pretraining and continued-pretraining.

This repository generates structured JSONL datasets for pretraining and
continued-pretraining data experiments. It focuses on reproducible synthetic signal
generation, schema validation, exact deduplication, duplicate reporting, and
Hugging Face dataset publishing.

The pipeline currently supports four signal families:

- `arithmetic` — integer arithmetic, word problems, comparisons, missing-value problems, and compact reasoning steps.
- `task_code` — beginner/intermediate Python tasks with short plans and code snippets.
- `educational_qa_mcq` — scenario-based multiple-choice questions with answer choices and explanations.
- `factual_restraint` — prompts that reward cautious answers and discourage unsupported claims.

![Architecture](docs/architecture.png)

---

## Current Status

The supported end-to-end pipeline is:

```text
generate -> validate -> dedup -> push_hf
```

Validated default posture:

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

- Groq-hosted Llama generation.
- JSON object generation contract: `{"items": [...]}`.
- Groq JSON object mode support.
- Groq Flex service-tier support.
- Exponential backoff with jitter for transient API failures.
- Per-batch diversity controls.
- Exact-only synthetic deduplication by default.
- Duplicate reporting by pipeline stage.
- Config-based `generate`, `validate`, `dedup`, and `push_hf` commands.
- Shared path resolution for `${DATA_DIR}/<run_name>`.
- Hugging Face push with `.env` token loading.
- Hugging Face dataset card generation.


---

## Supported Models

This project is validated with the following Groq models:

| Model | Use |
|---|---|
| `llama-3.1-8b-instant` | Recommended default for scalable synthetic generation. |
| `llama-3.3-70b-versatile` | Higher-quality option for smaller or quality-focused runs. |

Other models may work, but they are not currently validated for production-scale generation. The pipeline requires reliable JSON object output, strict schema following, and stable batched responses.

---

## Choosing a Run Size

Recommended progression:

| Target | Purpose |
|---:|---|
| `200000` | Smoke test after code or prompt changes. |
| `5000000` | Full pipeline validation. |
| `50000000` | Long-run stability test. |
| `600000000` | Production-scale generation target. |

---

## Generation Profiles

| Profile | Default Model | Runtime Posture | Purpose |
|---|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Higher concurrency, throughput-oriented settings. | Fast bulk generation when occasional retries are acceptable. |
| `balanced` | `llama-3.1-8b-instant` | Moderate concurrency, diversity controls, backoff. | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Lower concurrency, higher-quality model. | Smaller quality-focused runs or comparisons. |

`speed` and `balanced` intentionally use the same default model. The difference is runtime posture, not model family.

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
│   ├── COMMANDS.md
│   ├── DISK_SETUP.md
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

## Getting Started

### Prerequisites

- Python 3.12+
- Ubuntu 24.04 recommended
- Groq API key
- Hugging Face account and token, if publishing datasets

### Disk Setup

For larger runs, use a separate mounted data volume before cloning or generating data:

- [Disk setup guide](docs/DISK_SETUP.md)

If you are running a small smoke test on the boot disk, this step is optional.

### Installation

```bash
git clone https://github.com/tohio/slm-synthetic-data.git /data/slm-synthetic-data
cd /data/slm-synthetic-data

python3 -m venv venv
source venv/bin/activate

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

---

## Quickstart

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
```

Push to Hugging Face after verifying the deduped output:

```bash
make push
```

---

## Common Commands

Generate config:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Bootstrap output directories:

```bash
python bootstrap_dirs.py
```

Generate all signals:

```bash
make generate
```

Generate one signal:

```bash
make generate SIGNAL=educational_qa_mcq
```

Report exact duplicates:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```

Validate:

```bash
make validate
```

Deduplicate:

```bash
make dedup
```

Push to Hugging Face:

```bash
make push
```

See the full command reference:

- [Command reference](docs/COMMANDS.md)

---

## Pipeline Outputs

Each run writes to:

```text
data/runs/<run_name>/
```

Stages:

```text
raw/        generated JSONL records
validated/  schema-valid records
deduped/    exact-deduped records for downstream use
rejected/   rejected records and failed batches
manifests/  generated metadata, including Hugging Face dataset card
```

---

## Dataset Format

The pipeline writes one JSON object per line.

Example `arithmetic` record:

```json
{
  "type": "arithmetic",
  "question": "A box has 12 red pens and 9 blue pens. How many pens are in the box?",
  "steps": ["Add 12 and 9.", "12 + 9 = 21."],
  "answer": "21"
}
```

Each signal has its own schema. See:

- [Prompt documentation](prompts/README.md)
- [Package documentation](slm_synth/README.md)

---

## Deduplication Policy

Synthetic data uses exact deduplication by default:

```yaml
dedup:
  mode: "exact"
  enable_exact: true
  enable_fuzzy: false
  fuzzy_enabled: false
```

Fuzzy MinHash deduplication is intentionally disabled for synthetic data because useful examples can share structure, schemas, and wording patterns. Use fuzzy dedup only for explicit experiments.

---

## Hugging Face Publishing

`make push` uploads:

- deduped JSONL files
- generated dataset card as `README.md`

The Hugging Face repo target is configured in `configs/synthetic.yaml`:

```yaml
export:
  push_to_hf: true
  hf_repo: "tohio/slm-synthetic"
  private: false
```

The push stage loads credentials from `.env` using either:

```text
HF_TOKEN=...
```

or:

```text
HUGGINGFACE_HUB_TOKEN=...
```

---

## Resume After Interruption

If a run is interrupted, do not delete `data/` unless you intend to start over.

Inspect current counts:

```bash
for f in data/runs/*/raw/*.jsonl; do
  echo "$f $(wc -l < "$f")"
done
```

Resume an unfinished signal:

```bash
make generate SIGNAL=task_code
```

If a signal file is partial and you want a clean rerun for that signal:

```bash
rm -f data/runs/<run_name>/raw/task_code.jsonl
rm -f data/runs/<run_name>/rejected/task_code.jsonl
make generate SIGNAL=task_code
```

---

## Testing

Install test tools if needed:

```bash
pip install pytest
```

Run all tests:

```bash
make test
```

or:

```bash
python -m pytest -q
```

See:

- [Test guide](tests/README.md)

---

## Key Decisions

- **Use JSON objects, not top-level arrays.** Models are instructed to return `{"items": [...]}` because it works better with JSON object mode and batched generation.
- **Support only validated Groq Llama models for production runs.** The backend is configurable, but the pipeline depends on reliable schema-following behavior.
- **Use `llama-3.1-8b-instant` as the default bulk generator.** It is the recommended model for scalable synthetic generation.
- **Use `llama-3.3-70b-versatile` for quality-focused runs.** It is slower and more expensive, so it is better suited for smaller runs, audits, or comparisons.
- **Scale with moderate batch size plus concurrency.** `BATCH=4` with controlled concurrency is preferred over very large batches.
- **Use exact deduplication for synthetic data.** Fuzzy MinHash deduplication is disabled by default because synthetic examples intentionally share schemas and patterns.
- **Measure duplicate rate before scaling.** High exact duplicate rates indicate a generation-diversity issue and should be fixed before large runs.
- **Keep generated data out of git.** `data/` contains run outputs and should not be committed.


---

## Documentation

- [Command reference](docs/COMMANDS.md)
- [Disk setup](docs/DISK_SETUP.md)
- [Docs index](docs/README.md)
- [Configuration guide](configs/README.md)
- [Prompt guide](prompts/README.md)
- [Package guide](slm_synth/README.md)
- [Test guide](tests/README.md)

---

## Related Projects

- [slm](https://github.com/tohio/slm) — GPT-style LLM trained from scratch

---

## License

MIT


## Hugging Face Publishing

By default, `make push` publishes each signal to its own Hugging Face dataset repository:

```text
<namespace>/slm-synthetic-arithmetic
<namespace>/slm-synthetic-task-code
<namespace>/slm-synthetic-educational-qa-mcq
<namespace>/slm-synthetic-factual-restraint
```

Push a single signal with:

```bash
make push SIGNAL=arithmetic
```

Each repository receives one JSONL file and a public-facing dataset card.
