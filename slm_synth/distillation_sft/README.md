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
├── response_diversity.py # aggregate/per-signal exact response diversity
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

## Scaling

The production inventory contains 50 template families across 10 signals. Every
signal has at least four template families, and no template exceeds 30% of its
signal allocation. Debugging and factual-restraint prompts use substantive task,
code, claim, and safe-alternative combinations instead of varying only record
counts, fictional identifiers, or application labels.

| Accepted rows | Scaling posture |
| ---: | --- |
| 30,000 | Strong default target. Approximately 600–750 rows per template. |
| 50,000 | Strong. Approximately 1,000–1,250 rows per template. |
| 100,000 | Current design ceiling. Approximately 2,000–2,500 rows per template. |
| Above 100,000 | Review template concentration and response diversity before generation. |
| Above 150,000 | Add template families or external prompt sources first. |

Regression checks require zero exact or normalized prompt duplicates at 30,000
and 100,000 rows, at least four template families per signal, exactly 50 template
families overall, and a maximum 30% template share within each signal. Coverage
reports include aggregate and per-signal exact response-diversity statistics.
Publishing requires every signal to retain at least a 75% normalized exact-response ratio.
`DISTILLATION_SFT_MIN_UNIQUE_RESPONSE_RATIO` can override that threshold for an
explicitly reviewed dataset.

To scale beyond the current ceiling:

1. Add task structures within the affected signals rather than only new parameter values.
2. Keep new template families balanced within each signal.
3. Re-run the uniqueness and template-concentration checks at the proposed target and backfill range.
4. Run a paid smoke job and inspect teacher-response acceptance and quality before production.
