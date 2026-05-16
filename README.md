# SLM Synthetic Data

A modular synthetic-data engine for generating arithmetic, task-code, educational MCQ, and factual-restraint signals for SLM pretraining.

**Status:** Active development

This repository contains the synthetic-data generation pipeline used by the main SLM training project. It is designed to produce high-signal curriculum data that can be blended into the SLM pretraining corpus and reused across multiple model sizes.

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

The default target is intended to support the largest configured SLM model, currently assumed to be in the **1B–1.5B+** range.

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
    └── runs/<date>/
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
```

---

## Configuration

All generation settings are defined in:

```text
configs/synthetic.yaml
```

This file controls:

- Backend provider
- Backend model
- Sampling temperature
- Maximum tokens per sample
- Number of samples per signal type
- Output schema
- Validation rules
- Deduplication settings
- Total target token budget

Example settings include:

```yaml
target_tokens: 600000000

backend:
  provider: groq
  model: llama-3.3-70b-versatile

sampling:
  temperature: 0.7
  max_tokens: 2048
```

Adjusting the total dataset size should usually require changing only the target token budget and, if needed, the per-signal allocation.

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

### 1. Generate Synthetic Data

```bash
python -m slm_synth.generate --config configs/synthetic.yaml
```

Raw outputs are written to:

```text
data/runs/<date>/raw/
```

---

### 2. Validate Generated Data

```bash
python -m slm_synth.validate data/runs/<date>/raw/
```

Validated outputs are written to:

```text
data/runs/<date>/validated/
```

Rejected records are written to:

```text
data/runs/<date>/rejected/
```

---

### 3. Deduplicate

```bash
python -m slm_synth.dedup data/runs/<date>/validated/
```

Deduplicated outputs are written to:

```text
data/runs/<date>/deduped/
```

---

### 4. Export to Hugging Face

```bash
python -m slm_synth.push_hf data/runs/<date>/deduped/
```

The exported dataset can then be consumed by the main SLM curation pipeline.

---

## Integration with the Main SLM Project

The generated synthetic dataset plugs into the main SLM curation pipeline through the synthetic source modules:

```text
curator/sources/synthetic_arithmetic.py
curator/sources/synthetic_task_code.py
curator/sources/educational_qa_mcq.py
curator/sources/factual_restraint.py
```

The main SLM curator treats synthetic data like any other source:

- Downloaded or loaded
- Quality filtered
- Deduplicated
- Blended
- Tokenized
- Split into train and validation sets

Synthetic-source deficits should route to higher-signal fallback sources first.

Recommended fallback order:

```text
Synthetic source
  -> Nemotron Specialized
  -> FineWeb-Edu
  -> FineWeb
```

This keeps the final corpus from underfilling while preserving signal quality as much as possible.

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
pytest -q
```

Tests cover:

- Schema correctness
- Generation formatting
- Validation logic
- Deduplication behavior
- Backend integration boundaries

Recommended before large generation runs:

```bash
pytest -q
python -m slm_synth.generate --config configs/synthetic.yaml --dry-run
```

---

## Development Principles

This repository follows the same development approach as the main SLM project:

- Keep sources modular.
- Keep schemas explicit.
- Validate before deduplication.
- Preserve rejected records for debugging.
- Make every stage resumable.
- Avoid regenerating expensive data unnecessarily.
- Prefer small validation runs before scaling.
- Keep generated data compatible with the main SLM curation pipeline.

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

## License

TBD.

---

## Notes

This repository is intended to generate reusable synthetic pretraining data for SLM. The full dataset should be generated once for the largest target model and then subset during downstream curation for smaller models.

The synthetic dataset should not replace natural data sources. It should act as a compact curriculum layer inside the broader pretraining corpus.
