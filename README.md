# SLM Synthetic Data

This repository builds a reusable synthetic pretraining corpus for small language models up to approximately 1.5B parameters. The production path uses **grounded artifact generation** followed by **OpenRouter / DeepSeek rendering**.

## Qualified generation architecture

```text
deterministic grounded artifact generator
→ homogeneous batches of 32 grounded artifacts
→ deepseek/deepseek-v4-flash strict structured rendering
→ persisted batch manifest and resumable raw JSONL
→ validation → exact deduplication → Hugging Face publish
```

The design was qualified with single-record tests and batch-size testing. Batches of 32 were selected after repeated successful stress runs. Each final record is rooted in a deterministic, prevalidated artifact rather than a model-invented candidate.

## Synthetic signals

| Signal | Grounded source | Final stored record |
|---|---|---|
| `arithmetic` | Verified integer arithmetic backbone | `question`, `steps`, `answer` |
| `task_code` | Valid local Python function | `task`, `plan`, `code` |
| `educational_qa_mcq_math` | Verified math item with fixed choices/answer | `question`, `choices`, `correct_index`, `explanation` |
| `educational_qa_mcq_general` | Fixed evidence and answer relationship | `evidence`, `question`, `choices`, `correct_index`, `explanation` |
| `factual_restraint` | Controlled uncertainty/privacy/context scenario | `question`, `safe_answer` |

## Locked production corpus mix

| Signal | Estimated raw generation target |
|---|---:|
| `task_code` | 300.0M |
| `educational_qa_mcq_general` | 187.5M |
| `arithmetic` | 112.5M |
| `educational_qa_mcq_math` | 112.5M |
| `factual_restraint` | 50.0M |
| **Total** | **762.5M** |

The repository generates once and persists output for repeated downstream use. It intentionally accepts later validation/deduplication loss rather than planning top-up generation.

`TOKENS` is an estimated planning target converted into record counts through per-signal `avg_tokens_per_sample` values. This repository does **not** import the downstream training tokenizer; final training-token totals are measured when the corpus is tokenized for training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...       # only needed for publishing
```

## Quick smoke run

```bash
make configure TOKENS=100000 CONCURRENCY=4 RUN=grounded_quality_smoke
make preflight-artifacts
make generate
make report-artifacts
make validate
make dedup
make report-duplicates STAGE=deduped
make report-lengths STAGE=deduped
```

Inspect the generated records and length calibration before creating the production configuration:

```bash
make production-config CONCURRENCY=4
```

## Resume and telemetry

Generation persists every completed DeepSeek request under:

```text
data/runs/<run_name>/manifests/grounded/<signal>/batches/
```

Each manifest stores grounded artifacts, rendered records and request telemetry. Running `make generate` again resumes from completed batches and reconstructs the raw JSONL output without duplicate completed requests.

## Quality controls

- `make preflight-artifacts` walks the full configured artifact plan before paid rendering and rejects exact artifact duplicates or quality issues.
- Every batch is also checked immediately before model rendering for correctness, placeholders and schema-specific quality problems.
- `make report-artifacts` reports artifact-level exact duplicates, family coverage and structural variety.
- `make validate` validates final rendered records.
- `make dedup` applies exact deduplication to validated records.
- `make report-lengths` produces estimated per-record length calibration for target-row planning.

## Repository layout

```text
configs/                  Config generator and grounded template
slm_synth/artifacts/      Deterministic grounded artifact factories and preflight quality checks
slm_synth/grounded.py     Batch renderer, anchoring and atomic batch persistence
slm_synth/llm.py          OpenRouter/DeepSeek structured output client and retry telemetry
slm_synth/generate.py     Pipeline orchestration and bounded concurrency
slm_synth/report_artifacts.py  Artifact diversity/quality report
slm_synth/report_lengths.py    Per-signal size calibration report
slm_synth/validate.py     Raw-to-validated record validation
slm_synth/dedup.py        Exact deduplication
```

Full commands: [docs/COMMANDS.md](docs/COMMANDS.md)
