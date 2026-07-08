# `slm_synth/pretrain`

## Purpose

This package owns grounded synthetic pretraining record generation. It renders deterministic local artifacts into provider prompts, validates generated records, deduplicates outputs, writes manifests, and prepares public pretraining artifacts.

It does not produce SFT chat rows, DPO preference pairs, response-distillation rows, or model-training artifacts.

## Contents

```text
pretrain/
├── artifacts/              # deterministic grounded source objects
├── sources/                # renderers that turn artifacts into promptable records
├── generate.py             # live generation orchestration
├── grounded.py             # grounded batch rendering and persistence
├── validate.py             # raw-to-validated record validation
├── dedup.py                # exact deduplication
├── manifest.py             # run manifest and coverage outputs
├── preflight_artifacts.py  # source artifact quality checks
├── report_artifacts.py     # artifact coverage/quality reports
├── report_lengths.py       # per-record size estimation
└── push_hf.py              # Hugging Face publishing
```

## Key Files

| File | Purpose |
|---|---|
| `generate.py` | Coordinates pretraining generation, resume, validation, dedup, and reporting. |
| `grounded.py` | Builds grounded prompts and writes intermediate generation artifacts. |
| `schemas.py` | Shared record schemas for pretraining data flow. |
| `writer.py` | JSONL output helpers. |
| `repair.py` | Local repair helpers for recoverable malformed records. |

## How It Fits In

Pretraining outputs are consumed downstream as synthetic text records for continued pretraining or corpus mixing. Command usage is documented in `../../docs/COMMANDS.md`.

## Conventions

Pretraining signals are grounded in deterministic local artifacts before provider calls. Source records and internal metadata are used for validation and reporting; downstream public artifacts contain validated text records and separate manifests.
