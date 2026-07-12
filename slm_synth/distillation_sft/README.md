# `slm_synth/distillation_sft`

## Purpose

This package owns teacher response datasets for response distillation. It builds prompt records, calls the teacher through structured generation, applies prompt and response gates, writes per-signal public JSONL files, records manifests, reports coverage, and publishes `distillation-sft-*` artifacts.

It does not create DPO pairs, sample student outputs, train models, export models, or create logits artifacts.

## Contents

```text
distillation_sft/
├── signals.py          # supported distillation signal names
├── prompts.py          # prompt-record validation
├── public_metadata.py  # public category/template/eval metadata mapping
├── seeds.py            # smoke seed prompts
├── spec_builders.py    # production prompt-spec builders
├── prompt_quality.py   # duplicate/near-duplicate prompt preflight
├── batches.py          # teacher batch prompt and response contract
├── generation.py       # one-signal teacher generation
├── response_quality.py # lightweight response gates
├── orchestration.py    # multi-signal smoke and production runs
├── schema.py           # public row and teacher-output validation
├── io.py               # JSONL and manifest writers
├── report.py           # coverage reporting
├── card.py             # dataset card rendering
├── push_hf.py          # Hugging Face publishing
└── cli.py              # command-line entrypoint
```

## How It Fits In

Public artifacts use the `distillation-sft-*` naming surface. The Python package uses an underscore because Python packages cannot use hyphens. Downstream `slm-distillation` consumes the published response datasets.

## Conventions

Public rows have this contract:

```json
{
  "id": "string",
  "prompt": "string",
  "reasoning": null,
  "response": "string",
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null"
  }
}
```

Teacher/provider/run/cost/retry details stay in manifests. Public rows always use `reasoning: null`.
