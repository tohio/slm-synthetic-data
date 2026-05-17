# Configuration Files

This directory contains configuration files for the SLM synthetic-data generation pipeline.

The main configuration file is:

```text
configs/synthetic.yaml
```

This file controls:

- backend provider and model selection
- sampling parameters
- batch size and request concurrency
- total synthetic token budget
- rate-limit behavior
- signal-family allocation
- validation behavior
- deduplication behavior
- export behavior

Model-dependent values should not be edited manually. Use the configuration helper instead.

---

# `configure_synthetic.py`

`configs/configure_synthetic.py` updates the existing `configs/synthetic.yaml` file in place.

It automatically adjusts model-dependent fields based on:

- the selected Groq model
- the inferred model size
- nearest-bucket profile mapping
- recommended batch size
- recommended max output tokens
- recommended temperature
- recommended top-p
- recommended request concurrency
- the requested target token budget

This keeps the synthetic pipeline configuration valid, reproducible, and aligned with the selected Groq model.

---

## Default Model

If no model is provided, the Makefile uses:

```text
openai/gpt-oss-20b
```

This model is used as the default balance point for generation quality and throughput.

---

## Main Configuration File

The main config file is:

```text
configs/synthetic.yaml
```

The current default values are intended for a small smoke-test run:

```yaml
target_total_tokens: 200000

backend:
  provider: "groq"
  model: "openai/gpt-oss-20b"
  max_tokens: 512
  temperature: 0.4
  top_p: 0.95
  parallel_requests: 32

rate_limit:
  max_concurrency: 2

generation:
  batch_size: 32
```

For larger runs, update the config through `make configure`.

---

## Usage

### Configure with the default model

```bash
make configure TOKENS=600000000
```

This updates `configs/synthetic.yaml` using the default model:

```text
openai/gpt-oss-20b
```

---

### Configure with a specific Groq model

```bash
make configure \
  MODEL=llama-3.3-70b-versatile \
  TOKENS=300000000
```

---

### Override batch size

```bash
make configure \
  MODEL=llama-3.1-8b-instant \
  TOKENS=200000 \
  BATCH=8
```

---

## Direct Script Usage

The helper can also be called directly:

```bash
python configs/configure_synthetic.py \
  --model openai/gpt-oss-20b \
  --tokens 200000 \
  --config configs/synthetic.yaml
```

With a batch-size override:

```bash
python configs/configure_synthetic.py \
  --model llama-3.1-8b-instant \
  --tokens 200000 \
  --batch-size 8 \
  --config configs/synthetic.yaml
```

---

## Model Discovery

The helper validates the requested model against the list of models available from Groq.

It reads the Groq API key from `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

You can list available Groq models with:

```bash
make list-models
```

Example output:

```text
Available Groq Models:
 - llama-3.1-8b-instant
 - llama-3.3-70b-versatile
 - openai/gpt-oss-20b
 - openai/gpt-oss-120b
```

---

## Fields Updated by the Helper

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

All other configuration sections remain unchanged.

---

## Model Profile Mapping

The helper infers model size from the model name.

Examples:

```text
llama-3.1-8b-instant      -> 8B profile
openai/gpt-oss-20b        -> 20B profile
llama-3.3-70b-versatile   -> 70B profile
openai/gpt-oss-120b       -> extra-large fallback profile
```

Current profile behavior:

| Model Size | Batch Size | Max Tokens | Temperature | Top-p | Concurrency |
|---:|---:|---:|---:|---:|---:|
| `<= 10B` | 16 | 384 | 0.3 | 0.90 | 4 |
| `<= 30B` | 32 | 512 | 0.4 | 0.95 | 2 |
| `<= 80B` | 64 | 768 | 0.2 | 0.95 | 1 |
| `> 80B` | 64 | 768 | 0.2 | 0.95 | 1 |

For models larger than 80B, the helper uses the conservative 70B-style profile.

---

## Signal Allocation

Signal-family allocation is controlled by the `mix` section:

```yaml
mix:
  arithmetic:
    share: 0.30
    prompt_file: "prompts/arithmetic.yaml"
    source_module: "slm_synth.sources.arithmetic"

  task_code:
    share: 0.30
    prompt_file: "prompts/task_code.yaml"
    source_module: "slm_synth.sources.task_code"

  educational_qa_mcq:
    share: 0.30
    prompt_file: "prompts/educational_qa_mcq.yaml"
    source_module: "slm_synth.sources.educational_qa_mcq"

  factual_restraint:
    share: 0.10
    prompt_file: "prompts/factual_restraint.yaml"
    source_module: "slm_synth.sources.factual_restraint"
```

The current default mix is:

| Signal | Share |
|---|---:|
| `arithmetic` | 30% |
| `task_code` | 30% |
| `educational_qa_mcq` | 30% |
| `factual_restraint` | 10% |

These values may be edited manually when intentionally rebalancing the curriculum.

---

## Output Location

The config defines:

```yaml
run_name: "synthetic"
output_dir: "${DATA_DIR}/${run_name}"
```

With the default `DATA_DIR=data`, outputs are written under:

```text
data/synthetic
```

If the pipeline code expands the run directory differently, keep this README aligned with the implementation.

---

## Export Settings

Hugging Face export is controlled by:

```yaml
export:
  push_to_hf: true
  hf_repo: "${HF_USERNAME}/${HF_REPO}"
  private: false
  include_manifests: true
```

Set these environment variables before running `make push`:

```bash
export HF_USERNAME=<your_hf_username>
export HF_REPO=<your_dataset_repo>
```

Or define them in `.env`.

---

## What Should Be Edited Manually?

Avoid manually editing model-dependent fields:

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

Use:

```bash
make configure TOKENS=<token_budget>
```

or:

```bash
make configure MODEL=<groq_model> TOKENS=<token_budget>
```

Manual edits are acceptable for:

- signal-family shares
- prompt file paths
- source module paths
- validation settings
- deduplication settings
- export settings
- output path settings

---

## Related Makefile Targets

### Configure the pipeline

```bash
make configure TOKENS=600000000
```

### Configure with a specific model

```bash
make configure \
  MODEL=llama-3.3-70b-versatile \
  TOKENS=300000000
```

### List available Groq models

```bash
make list-models
```

### Run generation

```bash
make generate
```

### Run validation

```bash
make validate
```

### Run deduplication

```bash
make dedup
```

### Push to Hugging Face

```bash
make push
```

---

## Summary

`configure_synthetic.py` ensures that `configs/synthetic.yaml` stays:

- valid
- reproducible
- optimized for the selected model profile
- consistent with Groq model availability
- aligned with the requested token budget

Use `make configure` before large generation runs.
