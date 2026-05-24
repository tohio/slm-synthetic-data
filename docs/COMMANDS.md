# SLM Synthetic Data — Command Reference

The pipeline generates grounded synthetic pretraining records with **OpenRouter** and the qualified renderer `deepseek/deepseek-v4-flash`.

## Architecture

```text
deterministic grounded artifact factory
→ signal-homogeneous request of 32 artifacts
→ one strict structured-output DeepSeek rendering request
→ atomic completed-batch manifest + resumable raw JSONL
→ validate → exact dedup → publish
```

All five signals use the grounded path:

- `arithmetic`
- `task_code`
- `educational_qa_mcq_math`
- `educational_qa_mcq_general`
- `factual_restraint`

## Environment

Create `.env` with:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

`HF_TOKEN` is required only when pushing data. Never commit `.env`.

## Variables

| Variable | Default | Description |
|---|---:|---|
| `PROFILE` | `balanced` | Runtime posture: `speed`, `balanced`, or `quality`. |
| `TOKENS` | `200000` | Estimated generation target; converted to row targets using calibrated per-signal averages. |
| `MODEL` | `deepseek/deepseek-v4-flash` | OpenRouter renderer override. DeepSeek is the qualified baseline. |
| `BATCH` | `32` | Grounded artifacts per request. Only `32` is qualified for production. |
| `CONCURRENCY` | profile default | Number of in-flight grounded batch requests. Defaults: speed `8`, balanced `4`, quality `2`. |
| `RUN` | generated | Optional run name override for a fresh output directory. |
| `HF_REPO` | environment/default | Optional Hugging Face destination override. |
| `SIGNAL` | unset | Restrict a command to one signal. |
| `STAGE` | `deduped` | Stage for duplicate/length reporting. |

## Configure

Small all-signal smoke run:

```bash
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_quality_smoke
```

Locked production-target configuration:

```bash
make production-config CONCURRENCY=4
```

This writes `configs/synthetic.yaml` with the fixed corpus mix:

| Signal | Estimated target at full production scale |
|---|---:|
| `task_code` | 300.0M |
| `educational_qa_mcq_general` | 187.5M |
| `arithmetic` | 112.5M |
| `educational_qa_mcq_math` | 112.5M |
| `factual_restraint` | 50.0M |
| **Total** | **762.5M** |

The repository deliberately does not import a downstream training tokenizer. `TOKENS` is an estimated planning unit used to derive record counts. The final training-token count is measured downstream when the corpus is tokenized for training.

## Preflight planned artifacts before paid rendering

```bash
make preflight-artifacts
```

This deterministically walks the configured grounded-artifact plan before any model requests are made. It fails on placeholder/quality problems or exact grounded-artifact duplicates and stores an artifact report plus a SQLite fingerprint index under `manifests/grounded/`.

## Generate and resume

```bash
make generate
```

Generate only one signal:

```bash
make generate SIGNAL=task_code
```

Completed requests are stored atomically under:

```text
data/runs/<run_name>/manifests/grounded/<signal>/batches/
```

Rerunning `make generate` safely resumes from persisted completed batches and does not repeat completed requests. Transport interruptions may be retried automatically; unresolved failures are written to `rejected/` and the run can be resumed.

## Validate artifact diversity before scaling

```bash
make report-artifacts
```

This reports grounded artifact totals, exact artifact duplicates, structure counts, family coverage, and placeholder/quality issues. It reads the persisted grounded artifacts, not only the model-rendered wording.

A production run must not proceed if this report finds exact artifact duplicates or quality issues.

## Validate and deduplicate rendered output

```bash
make validate
make dedup
make report-duplicates STAGE=deduped
```

Synthetic data uses exact-only deduplication by default. Fuzzy deduplication remains disabled because structural similarity is expected in controlled generated data.

## Calibrate estimated row sizing

After a clean smoke or validation run:

```bash
make report-lengths STAGE=deduped
```

The report estimates per-record sizes using serialized character length and writes:

```text
data/runs/<run_name>/manifests/length_report_deduped.json
```

Use the recommended per-signal values to adjust `avg_tokens_per_sample` before the full production configuration. This is planning calibration, not a downstream tokenizer replacement.

## Recommended workflow

```bash
# Small quality smoke test
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_quality_smoke
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped

# After reviewing records and calibrated averages, prepare the one-time large run
make production-config CONCURRENCY=4
make generate
```

## Monitoring

```bash
watch -n 10 'find data/runs -path "*/raw/*.jsonl" -printf "%p %s\n" | sort'
find data/runs -path '*/manifests/grounded/*/batches/*.json' | wc -l
```

Use `tmux` for long generation runs.
