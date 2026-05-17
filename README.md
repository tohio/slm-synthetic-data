# SLM Synthetic Data

A modular synthetic-data engine for generating arithmetic, task-code, educational MCQ, and factual-restraint signals for SLM pretraining.

This repository contains the synthetic-data generation pipeline used by the main SLM training project. It is designed to produce high-signal curriculum data that can be blended into the SLM pretraining corpus and reused across multiple model sizes.

![Architecture](docs/architecture.png)

---

## Overview

`slm-synthetic-data` implements a fully modular synthetic-data generation pipeline for the SLM project.

The pipeline generates structured synthetic examples across four signal families:

- `arithmetic` — multi-step numeric reasoning
- `task_code` — task decomposition, programming reasoning, and pseudocode
- `educational_qa_mcq` — educational multiple-choice questions with explanations
- `factual_restraint` — uncertainty handling, refusal behavior, and hallucination resistance

The goal is to provide compact, high-signal pretraining data that improves downstream SFT stability, strengthens reasoning behavior, and provides early priors for safer responses.

Unlike pipelines that generate separate synthetic datasets for each model size, this repository generates one reusable synthetic dataset sized for the largest target model. Smaller models consume subsets of the same dataset.

---

## Why Synthetic Data?

Synthetic curriculum data provides targeted training signals that are difficult to guarantee through web-scale data alone.

It helps introduce:

- Structured reasoning priors
- Educational comprehension priors
- Arithmetic and task decomposition priors
- Factual restraint and uncertainty handling
- High information density per token
- Low-cost scalable generation using Groq or other supported backends

Synthetic data is expected to represent approximately **3–7%** of the total pretraining corpus, but it can have an outsized impact on SFT efficiency, alignment behavior, and sanity-eval performance.

---

## Target Dataset Size

Dataset sizing is controlled by:

```text
configs/synthetic.yaml
```

The active token-budget field is:

```yaml
target_total_tokens: 200000
```

The default checked-in value is intentionally small and is suitable for smoke testing. For larger runs, update the token budget through the configuration helper.

Recommended synthetic token budget:

| Model Size | Synthetic Tokens Consumed |
|---:|---:|
| 125M | 60M–120M |
| 350M | 120M–200M |
| 1B | 180M–300M |
| 1.5B+ | Full dataset |

The full dataset target is typically **400M–600M tokens**.

Smaller models do not require separate synthetic generation runs. They consume a subset of the shared dataset during SLM curation and blending.

---

## Repository Structure

```text
slm-synthetic-data/
├── README.md
├── requirements.txt
├── .env.sample
├── Makefile
├── pytest.ini
│
├── configs/
│   ├── README.md
│   ├── configure_synthetic.py
│   └── synthetic.yaml
│
├── docs/
│   └── architecture.png
│
├── prompts/
│   ├── arithmetic.yaml
│   ├── task_code.yaml
│   ├── factual_restraint.yaml
│   └── educational_qa_mcq.yaml
│
├── slm_synth/
│   ├── schemas.py
│   ├── prompt_loader.py
│   ├── llm.py
│   ├── rate_limit.py
│   ├── generate.py
│   ├── validate.py
│   ├── dedup.py
│   ├── push_hf.py
│   └── sources/
│       ├── arithmetic.py
│       ├── task_code.py
│       ├── factual_restraint.py
│       └── educational_qa_mcq.py
│
├── tests/
│   ├── test_generate.py
│   ├── test_validate.py
│   ├── test_dedup.py
│   └── test_schemas.py
│
└── data/
    └── synthetic/
        ├── raw/
        ├── validated/
        ├── deduped/
        ├── rejected/
        └── manifests/
```

The layout mirrors the main SLM repository: modular, testable, resumable, and easy to extend.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<you>/slm-synthetic-data.git
cd slm-synthetic-data
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Create the environment file:

```bash
cp .env.sample .env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your backend credentials in `.env`.

Example:

```bash
GROQ_API_KEY=your_groq_api_key_here
HF_USERNAME=your_hf_username
HF_REPO=your_dataset_repo
```

Install Make if it is not already available:

```bash
sudo apt install -y make
```

---

## Configuration

All generation settings are defined in:

```text
configs/synthetic.yaml
```

This file controls:

- backend provider and model
- sampling parameters
- batch size and request concurrency
- total synthetic token budget
- rate-limit behavior
- signal-family allocation
- validation behavior
- deduplication behavior
- export behavior

Model-dependent values should normally be updated through the configuration helper instead of being edited manually.

The helper script is:

```text
configs/configure_synthetic.py
```

The helper updates the existing `configs/synthetic.yaml` file in place.

---

### Configure with the Default Model

The default model is:

```text
openai/gpt-oss-20b
```

Configure the token budget:

```bash
make configure TOKENS=600000000
```

This updates `configs/synthetic.yaml` using the default model and the model profile inferred by `configs/configure_synthetic.py`.

---

### Configure with a Specific Groq Model

```bash
make configure \
  MODEL=llama-3.3-70b-versatile \
  TOKENS=300000000
```

---

### Override Batch Size

```bash
make configure \
  MODEL=llama-3.1-8b-instant \
  TOKENS=200000 \
  BATCH=8
```

---

### List Available Groq Models

The helper validates the requested model against Groq using `GROQ_API_KEY`.

List available models:

```bash
make list-models
```

---

### Fields Updated by the Helper

The helper patches only model-dependent and budget-dependent fields:

```text
target_total_tokens
backend.provider
backend.model
backend.max_tokens
backend.temperature
backend.top_p
backend.parallel_requests
rate_limit.max_concurrency
generation.batch_size
```

Other configuration sections remain unchanged.

For more details, see:

```text
configs/README.md
```

---

## Pipeline Stages

The synthetic-data pipeline has four main stages:

1. **Generate**
   - Calls the configured LLM backend.
   - Produces raw JSONL records.
   - Writes records by signal family.

2. **Validate**
   - Enforces schemas.
   - Rejects malformed records.
   - Checks required fields.
   - Writes accepted and rejected outputs.

3. **Deduplicate**
   - Removes exact duplicates.
   - Optionally applies fuzzy or similarity-based deduplication.
   - Preserves useful curriculum variation where appropriate.

4. **Export**
   - Writes final dataset artifacts.
   - Optionally pushes the dataset to Hugging Face.

---

## Running the Pipeline

The pipeline is normally run through the Makefile.

The default config is:

```text
configs/synthetic.yaml
```

Supported signals:

- `arithmetic`
- `task_code`
- `educational_qa_mcq`
- `factual_restraint`

---

### 1. Configure the Pipeline

Before a large run, update the config for the target token budget and Groq model:

```bash
make configure TOKENS=600000000
```

For a small smoke test:

```bash
make configure TOKENS=200000
```

For a specific model:

```bash
make configure \
  MODEL=llama-3.3-70b-versatile \
  TOKENS=300000000
```

---

### 2. Generate Synthetic Data

Generate all signals:

```bash
make generate
```

Generate only one signal:

```bash
make generate SIGNAL=arithmetic
```

Raw outputs are written under the configured output directory.

With the default config, the output directory is derived from:

```yaml
run_name: "synthetic"
output_dir: "${DATA_DIR}/${run_name}"
```

With `DATA_DIR=data`, this resolves to:

```text
data/synthetic/
```

---

### 3. Validate Generated Data

```bash
make validate
```

Validated records are written under the configured output directory.

---

### 4. Deduplicate

```bash
make dedup
```

Deduplicated records are written under the configured output directory.

---

### 5. Export to Hugging Face

```bash
make push
```

The exported dataset can then be consumed by the main SLM curation pipeline.

The Hugging Face target is controlled by:

```yaml
export:
  push_to_hf: true
  hf_repo: "${HF_USERNAME}/${HF_REPO}"
  private: false
  include_manifests: true
```

Set these values in `.env` or export them in the shell:

```bash
export HF_USERNAME=<your_hf_username>
export HF_REPO=<your_dataset_repo>
```

---

### Run the Full Pipeline

```bash
make all
```

This runs:

```text
generate -> validate -> dedup -> push
```

---

## Signal Families

### Arithmetic

The arithmetic source generates multi-step numeric reasoning examples.

Examples may include:

- Word problems
- Multi-step arithmetic
- Unit-style reasoning
- Equation-based reasoning
- Short explanations

Purpose:

- Improve numeric reasoning priors
- Strengthen structured step-by-step reasoning
- Reduce brittleness on basic arithmetic sanity checks

---

### Task Code

The task-code source generates examples focused on programming and procedural reasoning.

Examples may include:

- Task decomposition
- Pseudocode
- Simple implementation plans
- Input/output reasoning
- Debugging-style explanations

Purpose:

- Improve code-adjacent reasoning
- Strengthen decomposition behavior
- Support downstream code SFT stability

---

### Educational QA MCQ

The educational MCQ source generates multiple-choice questions with explanations.

Examples may include:

- Question
- Answer choices
- Correct answer
- Explanation
- Subject or difficulty metadata

Purpose:

- Improve educational QA behavior
- Add compact factual and conceptual signal
- Strengthen explanation quality

---

### Factual Restraint

The factual-restraint source generates examples where the model should avoid unsupported claims.

Examples may include:

- Ambiguous factual questions
- Questions requiring uncertainty
- Refusal or caveat behavior
- “I don’t know” style responses
- Hallucination-resistant answer patterns

Purpose:

- Reduce unsupported factual claims
- Improve uncertainty handling
- Strengthen safe-response priors before SFT and DPO

---

## Testing

Run the full test suite:

```bash
make test
```

Tests cover:

- Schema correctness
- Generation formatting
- Validation logic
- Deduplication behavior
- Backend integration boundaries

Recommended smoke test before a large generation run:

```bash
make configure TOKENS=200000
make test
make generate SIGNAL=arithmetic
make validate
make dedup
```

Use a small signal run first to confirm credentials, prompt loading, schema validation, and backend connectivity before scaling to a full generation run.

---

## Roadmap

Planned improvements:

- Difficulty-adaptive arithmetic curriculum
- Multi-stage task decomposition
- Better MCQ distractor generation
- Adversarial factual-restraint generation
- Multi-backend support
  - Groq
  - Local models
  - Hugging Face Inference
- Dataset cards
- Dataset-level metrics
- Per-signal quality reports
- Token distribution reports
- Cost and throughput tracking

---

## Related Projects

- [slm](https://github.com/tohio/slm) — GPT-style LLM trained from scratch — data curation, pretraining, SFT, and DPO alignment

---

## License

MIT.
