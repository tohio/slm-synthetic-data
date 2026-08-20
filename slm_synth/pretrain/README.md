# `slm_synth/pretrain`

## Purpose

This package owns grounded synthetic pretraining record generation. It renders deterministic local artifacts into provider prompts, validates generated records, deduplicates outputs, writes manifests, and prepares public pretraining artifacts.

It does not produce SFT chat rows, DPO preference pairs, response-distillation rows, or model-training artifacts.

## Contents

```text
pretrain/
├── artifacts/              # deterministic grounded source objects
├── generate.py             # live generation orchestration
├── grounded.py             # grounded batch rendering and persistence
├── validate.py             # raw-to-validated record validation
├── dedup.py                # global exact/near deduplication and consolidation
├── manifest.py             # run manifest and coverage outputs
├── preflight_artifacts.py  # source artifact quality checks
├── report_artifacts.py     # artifact coverage/quality reports
├── report_diversity.py     # bounded template/near-duplicate diversity audit
├── report_lengths.py       # per-record size estimation
└── push_hf.py              # quality-gated consolidated Hugging Face publishing
```

## Key Files

| File | Purpose |
|---|---|
| `generate.py` | Coordinates pretraining generation, resume, validation, dedup, and reporting. |
| `grounded.py` | Builds grounded prompts and writes intermediate generation artifacts. |
| `schemas.py` | Shared record schemas for pretraining data flow. |
| `writer.py` | JSONL output helpers. |

## How It Fits In

Pretraining outputs are consumed downstream as synthetic text records for continued pretraining or corpus mixing. Command usage is documented in `../../docs/COMMANDS.md`.

## Conventions

Pretraining signals are grounded in deterministic local artifacts before provider calls. `grounded.py` renders artifacts into structured provider prompts, validates rendered records, and persists batch manifests for resume/reporting. Deduplication is global across every signal and writes one public `deduped/pretrain.jsonl` file with `id`, `text`, and `metadata.signal`. Exact or structural near-duplicates are rejected.

`curate.py` owns completion accounting. It counts estimated tokens only from
validated, globally unique public text. Rejected candidates never count. A
deficit causes another round using unused candidate indexes while preserving the
configured signal shares. The run completes only when every signal reaches its
accepted-token target; candidate-inventory or cost exhaustion produces an
explicit shortfall report and a nonzero exit.

Arithmetic preflight additionally requires every planned source artifact to
have a distinct structure before rendering. Arithmetic questions preserve
their assigned reasoning family, semantic context, numeric facts, and verified
local answer. Its 288-candidate quality capacity is a ceiling: larger token
plans do not pad the signal with operand-only variants.

`task_code` uses a finite 24-algorithm catalog. Public tasks are deterministic
from validated local code, while the teacher generates only a short faithful
plan. Renamed functions, field substitutions, and threshold-only changes do
not count as additional candidates.

`educational_qa_mcq_math` uses a finite 24-relationship catalog. Questions,
choices, answers, and verification expressions are authoritative local data;
the teacher supplies only the explanation. Its capacity is a hard ceiling, and
preflight rejects repeated source structures instead of accepting numeric
variations as additional candidates.

`educational_qa_mcq_general` uses one locally grounded candidate for each of 24
distinct reasoning families. Evidence, questions, choices, and answers remain
authoritative local data; the teacher supplies only the explanation. Slot
substitutions such as changed names, objects, places, or counts do not increase
the signal's candidate capacity.

`factual_restraint` uses a finite 32-scenario catalog spanning uncertainty,
ambiguity, privacy, unannounced information, rumors, and medical, legal, and
financial decisions. Questions and behavior requirements remain local, while
the teacher supplies only the natural answer. Entity, date, location, and
amount substitutions do not increase candidate capacity.

`make pretrain-report` writes `manifests/diversity_report_<stage>.json`. The
report uses deterministic bounded sampling to measure normalized template
reuse, near-duplicate clusters, artifact-family concentration, and exact
template overlap across signals. For the deduped stage it is a blocking quality
gate, and publishing independently audits the entire consolidated file before
creating a remote commit.
