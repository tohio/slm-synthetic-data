# SLM Synthetic Data — Command Reference

Complete command reference for the synthetic data generation pipeline. For the project overview, installation notes, and architecture, see the top-level `README.md`.

---

## Variables

Most Makefile targets accept these variables as overrides.

| Variable | Default | Description |
|---|---:|---|
| `PROFILE` | `balanced` | Generation profile. One of `speed`, `balanced`, or `quality`. |
| `TOKENS` | required for `configure` | Target token budget for the run. Example: `50000000` for 50M. |
| `BATCH` | profile/signal default | Batch size override. `BATCH=4` is the recommended default for scalable runs. |
| `CONCURRENCY` | profile default | Number of concurrent Groq requests. `8` is the recommended balanced setting. |
| `SERVICE_TIER` | `flex` | Groq service tier. One of `flex`, `default`, or `auto`. |
| `MODEL` | profile default | Groq model override. Only `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are validated. |
| `SIGNAL` | unset | Optional single-signal override for `make generate`. Values: `arithmetic`, `task_code`, `educational_qa_mcq_math`, `educational_qa_mcq_general`, `factual_restraint`. |
| `DATA_DIR` | resolver default | Root location for generated run outputs when `${DATA_DIR}` appears in config. |

---

## Supported Models

This project is validated with two Groq models:

| Model | Use |
|---|---|
| `llama-3.1-8b-instant` | Recommended default for scalable non-MCQ bulk generation. |
| `llama-3.3-70b-versatile` | Default for `educational_qa_mcq_math` and `educational_qa_mcq_general`; also suitable for quality-focused comparisons. |

Other models may work, but they are not validated for production generation. The pipeline depends on reliable JSON object output, strict schema following, and stable batched generation.

---

## Profiles

| Profile | Default Model | Runtime Posture | Purpose |
|---|---|---|---|
| `speed` | `llama-3.1-8b-instant` | Higher concurrency / throughput-oriented defaults. | Fast bulk generation when retry/backoff behavior is acceptable. |
| `balanced` | `llama-3.1-8b-instant` | Moderate concurrency, diversity controls, JSON object mode, and backoff. | Recommended default for scalable runs. |
| `quality` | `llama-3.3-70b-versatile` | Lower concurrency, higher-cost model. | Smaller quality-focused runs or comparison runs. |

`speed` and `balanced` intentionally use the same backend default model. The difference is runtime posture, not model family. MCQ signals are configured with a signal-level `llama-3.3-70b-versatile` override in `configs/synthetic_template.yaml`.

---

## One-Time Setup

### `make install`

Install dependencies if the Makefile provides this target.

```bash
make install
```

If dependencies are managed manually:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Environment file

Create a `.env` file with the credentials needed by the pipeline.

```bash
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
```

`HUGGINGFACE_HUB_TOKEN` may be used instead of `HF_TOKEN`.

Never commit `.env`.

---

## Pipeline Overview

The standard pipeline is:

```text
configure -> bootstrap -> generate -> duplicate report -> validate -> duplicate report -> dedup -> duplicate report -> push
```

Outputs are written under:

```text
data/runs/<run_name>/
├── raw/
├── validated/
├── deduped/
├── rejected/
└── manifests/
```

Use `deduped/*.jsonl` as the default downstream training input.

---

## Configuration

### `make configure`

Generate `configs/synthetic.yaml` from the template and CLI overrides.

```bash
make configure PROFILE=balanced TOKENS=50000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Common examples:

```bash
# Small smoke run
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex

# 5M validation run
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex

# 50M scale check
make configure PROFILE=balanced TOKENS=50000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex

# Quality-focused run
make configure PROFILE=quality TOKENS=5000000 BATCH=4 CONCURRENCY=4 SERVICE_TIER=flex
```

Optional model override:

```bash
make configure PROFILE=balanced MODEL=llama-3.1-8b-instant TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
```

Unvalidated model names print a warning. The warning is non-blocking.

---

## Bootstrap

### `python bootstrap_dirs.py`

Create the directory tree for the configured run.

```bash
python bootstrap_dirs.py
```

This resolves `${DATA_DIR}` consistently across pipeline stages.

---

## Generation

### `make generate`

Generate all configured synthetic signals.

```bash
make generate
```

The generator uses JSON object output, diversity controls, request backoff, and incremental JSONL writes.

### Generate one signal

```bash
make generate SIGNAL=arithmetic
make generate SIGNAL=task_code
make generate SIGNAL=educational_qa_mcq_math
make generate SIGNAL=educational_qa_mcq_general
make generate SIGNAL=factual_restraint
```

Use signal-specific generation when resuming after interruption or when testing one prompt.

### Live output

For long runs, prefer direct execution over piping through `tee` if you want immediate progress output:

```bash
export PYTHONUNBUFFERED=1
make generate
```

When output is piped through `tee`, Python output may be buffered.

---

## Duplicate Reporting

### `python -m slm_synth.report_duplicates`

Report exact duplicate counts and malformed JSON counts for a stage.

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage validated
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

Healthy synthetic generation should keep exact duplicates low before dedup.

| Duplicate Rate | Interpretation |
|---:|---|
| `0–5%` | Healthy. |
| `5–15%` | Inspect prompt diversity. |
| `15%+` | Do not scale until generation diversity is improved. |
| `50%+` | Severe generation collapse. |

---

## Validation

### `make validate`

Validate generated JSONL records against signal schemas.

```bash
make validate
```

Writes accepted records to `data/runs/<run_name>/validated/` and rejected records to `data/runs/<run_name>/rejected/`.

Check counts after validation:

```bash
for f in data/runs/*/validated/*.jsonl; do
  echo "$(basename "$f") $(wc -l < "$f")"
done
```

---

## Deduplication

### `make dedup`

Deduplicate validated records.

```bash
make dedup
```

For synthetic data, the default policy is exact-only deduplication:

```yaml
dedup:
  mode: exact
  enable_exact: true
  enable_fuzzy: false
```

Fuzzy MinHash dedup is not recommended for synthetic data because generated records intentionally share structure and schema. Use fuzzy dedup only for explicit experiments.

Clean deduped output before rerunning dedup:

```bash
python - <<'CLEAN_DEDUP'
import yaml
from slm_synth.paths import resolve_output_dir

cfg = yaml.safe_load(open("configs/synthetic.yaml"))
deduped = resolve_output_dir(cfg) / "deduped"
for p in deduped.glob("*.jsonl"):
    p.unlink()
    print("removed", p)
CLEAN_DEDUP
```

Then rerun:

```bash
make dedup
```

---

## Hugging Face Push

### `make push`

Upload deduped JSONL files and the generated dataset card to Hugging Face.

```bash
make push
```

The push step reads credentials from `.env`:

```bash
HF_TOKEN=hf_...
```

or:

```bash
HUGGINGFACE_HUB_TOKEN=hf_...
```

The target repo is configured in `configs/synthetic.yaml`.

Expected uploads:

```text
README.md
arithmetic.jsonl
task_code.jsonl
educational_qa_mcq_math.jsonl
educational_qa_mcq_general.jsonl
factual_restraint.jsonl
```

---

## Common Runs

### Smoke run

```bash
export PYTHONUNBUFFERED=1
make configure PROFILE=balanced TOKENS=200000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
make dedup
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

### 5M validation run

```bash
export PYTHONUNBUFFERED=1
make configure PROFILE=balanced TOKENS=5000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage validated
make dedup
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

### 50M scale check

```bash
export PYTHONUNBUFFERED=1
make configure PROFILE=balanced TOKENS=50000000 BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
rm -rf data
python bootstrap_dirs.py
make generate
```

After generation completes:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
make validate
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage validated
make dedup
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage deduped
```

### Push a completed run

```bash
make push
```

---

## Resume After Interruption

Do not remove `data/` if you want to keep current outputs.

Check current run and raw counts:

```bash
grep -n "run_name\|output_dir" configs/synthetic.yaml

for f in data/runs/*/raw/*.jsonl; do
  echo "$f $(wc -l < "$f")"
done
```

If a signal is incomplete and you want to rerun it cleanly, remove only that signal's files:

```bash
rm -f data/runs/*/raw/arithmetic.jsonl data/runs/*/rejected/arithmetic.jsonl
make generate SIGNAL=arithmetic
```

Then continue with remaining signals:

```bash
make generate SIGNAL=task_code
make generate SIGNAL=educational_qa_mcq_math
make generate SIGNAL=educational_qa_mcq_general
make generate SIGNAL=factual_restraint
```

---

## Monitoring Long Runs

In another shell, monitor raw files:

```bash
watch -n 10 'find data/runs -path "*/raw/*.jsonl" -printf "%p %s\n" | sort'
```

Check active Python processes:

```bash
ps -ef | grep -E "slm_synth.generate|python -m" | grep -v grep
```

Use `tmux` for long runs:

```bash
tmux new -s synth50m
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t synth50m
```

---

## Troubleshooting

### `make: python: No such file or directory`

The virtual environment is not active or the Makefile expects `python` on PATH.

```bash
source .venv/bin/activate || source venv/bin/activate
which python
python --version
```

### Output appears stuck when using `tee`

Python output may be buffered when piped.

```bash
export PYTHONUNBUFFERED=1
make generate
```

### Flex capacity exceeded

Temporary capacity errors are expected on Groq Flex. The pipeline uses exponential backoff and jitter. If rejected batches increase, lower concurrency:

```bash
make configure PROFILE=balanced TOKENS=50000000 BATCH=4 CONCURRENCY=6 SERVICE_TIER=flex
```

### High duplicate rate

Run duplicate reporting before scaling:

```bash
python -m slm_synth.report_duplicates --config configs/synthetic.yaml --stage raw
```

If duplicate rate exceeds 15%, improve signal diversity before larger runs.

---

## Suggested Scaling Path

Use staged scale-up instead of jumping directly to very large runs:

```text
200K smoke run -> 5M validation run -> 50M scale check -> larger production run
```

Before moving to the next scale, confirm:

```text
rejected_batches near 0
bad_json = 0
duplicate rate under 5%
validation rejects under 1%
dedup retention above 95%
```


## Hugging Face Push

Default push behavior publishes each signal to a separate Hugging Face dataset repository:

```bash
make push
```

Push only one signal:

```bash
make push SIGNAL=arithmetic
make push SIGNAL=task_code
make push SIGNAL=educational_qa_mcq_math
make push SIGNAL=educational_qa_mcq_general
make push SIGNAL=factual_restraint
```
