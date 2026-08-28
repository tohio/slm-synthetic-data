# `slm_synth/distillation_sft`

## Purpose

Generate teacher prompt/response data for response distillation.

This package is optimized for **high-quality, varied teacher responses to student-appropriate prompts**. It owns its own prompt/response novelty checks because repeated generic answers are especially damaging for response distillation.

## Contents

~~~text
distillation_sft/
├── pipeline.py              # supported production path
├── signals.py               # ten distillation signal names
├── schema.py                # public row contract
├── public_metadata.py       # public taxonomy metadata
├── prompt_quality.py        # prompt normalization used by publication checks
├── response_quality.py      # response quality/publication checks
├── response_diversity.py    # response cluster/diversity reporting
├── io.py                    # public dataset/manifest writers
├── report.py                # coverage reporting
├── card.py                  # dataset-card generation
├── push_hf.py               # consolidated HF publication
└── cli.py                   # report/card CLI
~~~

## Production Flow

~~~text
semantic derivation
→ student-appropriate prompt
→ prompt novelty
→ teacher response
→ deterministic row/response validation
→ response novelty
→ Nemotron judge
→ Gemma reviewer
→ prompt/response exact dedup
→ datasets/<signal>.jsonl
~~~

Default models:

| Role | Model |
|---|---|
| derivation | `openai/gpt-5.6-luna-pro` |
| task | `deepseek/deepseek-v4-flash` |
| teacher response | `deepseek/deepseek-v4-flash` |
| judge | `nvidia/nemotron-3.5-lightning` |
| reviewer | `google/gemma-4-31b-it` |

## How It Fits In

The public dataset is consumed by `slm-distillation`. Student sampling and model training are not performed here.

See [Architecture](../../docs/ARCHITECTURE.md).
For every supported override, see [Distillation-SFT parameters](../../docs/PARAMETERS.md#distillation-sft-parameters).

## Usage

~~~bash
make distillation-sft-smoke

DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
make distillation-sft-generate

DISTILLATION_SFT_INSPECT_RUN=distillation-sft-production-001 \
make distillation-sft-inspect

DISTILLATION_SFT_REPORT_RUN=distillation-sft-production-001 \
make distillation-sft-report

DISTILLATION_SFT_PUSH_RUN=distillation-sft-production-001 \
make distillation-sft-push
~~~

Focused smoke:

~~~bash
DISTILLATION_SFT_SIGNALS=code make distillation-sft-smoke
~~~

## Public Contract

~~~json
{
  "id": "...",
  "prompt": "...",
  "reasoning": null,
  "response": "...",
  "metadata": {}
}
~~~

The `reasoning` field is fixed to `null`. Private chain-of-thought, provider data, retry/cost data, and generation-only lineage are not public training content.

## Outputs

~~~text
data/distillation/runs/<run>/
├── datasets/<signal>.jsonl
├── manifests/
├── internal/<signal>/
├── coverage.json
└── README.md
~~~

## Conventions

- Prompts must be answerable from their own supplied information.
- Do not create fake dependencies on absent files/attachments/context.
- Response exact/near-duplicate checks occur before judge.
- Final dedup rejects duplicate prompt+response, duplicate prompt, and duplicate response.
- Reviewer contradiction/calibration checks remain dataset-specific.
- Manual post-run adjudication is not part of the supported production path.
