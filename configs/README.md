# Configuration

This directory contains configuration tools for synthetic generation.

## Files

| File | Purpose |
|---|---|
| `configure_synthetic.py` | Generates `configs/synthetic.yaml` from profile and CLI arguments. |
| `synthetic_template.yaml` | Template used by the config generator. |
| `synthetic.yaml` | Generated run configuration. This file is usually local/run-specific. |

## Common commands

Generate a small smoke-test config:

```bash
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Generate a 5M-token pipeline test config:

```bash
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Set the Hugging Face repository target:

```bash
make configure PROFILE=balanced TOKENS=5000000 HF_REPO=tohio/slm-synthetic
```

## Profiles

Profiles set defaults for model choice, generation temperature, max-token headroom, concurrency, and service tier.

| Profile | Default model | Runtime posture | Use |
|---|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Higher concurrency, throughput-oriented settings | Fastest bulk generation. |
| `balanced` | `llama-3.1-8b-instant` | Moderate concurrency, diversity controls, backoff | Recommended default. |
| `quality` | `llama-3.3-70b-versatile` | Lower concurrency, higher-quality model | Smaller quality-focused runs. |

`speed` and `balanced` intentionally use `llama-3.1-8b-instant` for both candidate and response passes. The difference is runtime posture, not model family. Each signal exposes `candidate_model` and `response_model` for explicit experiments.

## Supported models

Validated models:

- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`

Other models may be used for experiments, but they are not validated for production-scale generation. The pipeline requires reliable JSON object output and schema adherence.

## Important fields

| Field | Description |
|---|---|
| `output_dir` | Run output directory. Defaults to `${DATA_DIR}/<run_id>`. |
| `target_total_tokens` | Target token budget used to derive per-signal sample counts. The MCQ allocation is split 25% math / 75% general within its original share. |
| `backend.model` | Default Groq model used when a role-specific override is not provided. |
| `mix.<signal>.candidate_model` | Model used to author unanswered candidates. |
| `mix.<signal>.response_model` | Model used to answer or complete candidates into final raw records. |
| `backend.json_mode` | Enables JSON object output. Should remain enabled. |
| `backend.service_tier` | Groq service tier, commonly `flex` for high-throughput runs. |
| `backend.parallel_requests` | Concurrent request count. Scale this carefully. |
| `generation.diversity.enabled` | Enables per-batch diversity context. Should remain enabled. |
| `dedup.mode` | Should be `exact` for synthetic data. |
| `hf.repo_id` | Hugging Face dataset repository. |

## Path resolution

The config uses `${DATA_DIR}` by default:

```yaml
output_dir: "${DATA_DIR}/<run_id>"
```

If `DATA_DIR` is not exported, the shared resolver defaults to:

```text
data/runs/<run_id>
```
