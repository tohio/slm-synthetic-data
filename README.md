# SLM Synthetic Data

Synthetic dataset generation for the SLM training stack.

## Overview

This repository produces exactly five synthetic dataset products:

1. pretraining
2. generic SFT
3. generic DPO
4. distillation SFT
5. distillation DPO

It owns generation, deterministic validation, model-based quality review, manifests, reporting, and Hugging Face publication. It does not train student models, run model evaluation, create checkpoints, export models, or generate logits.

## Architecture

All five products use one shared mechanical runtime under `slm_synth/runtime/`. Dataset packages own their prompts, schemas, deterministic validators, judge/reviewer semantics, final acceptance rules, reports, and publication contracts.

```text
slm_synth/runtime/
  backend.py      OpenRouter backend construction and routing
  batching.py     batching and cardinality helpers
  stages.py       concurrent stage execution, retries, recursive isolation
  novelty.py      exact/near-duplicate filtering
  io.py           JSON/JSONL helpers
  reporting.py    shared report evidence/token helpers

slm_synth/
  pretrain/
  sft/
  dpo/
  distillation_sft/
  distillation_dpo/
```

Supported production flows:

```text
Pretrain:
grounded generation -> deterministic validation -> Gemma judge -> Luna reviewer
-> final global dedup -> accepted-token accounting/backfill -> publish

SFT:
derivation -> task -> novelty -> answer -> deterministic validation
-> Nemotron judge -> Gemma reviewer -> final dedup -> publish

DPO:
derivation -> task -> novelty -> pair -> deterministic validation
-> Nemotron judge -> Gemma reviewer -> final dedup -> publish

Distillation SFT:
derivation -> student prompt -> novelty -> teacher response
-> deterministic validation/response novelty -> Nemotron judge
-> Gemma reviewer -> final prompt/response dedup -> publish

Distillation DPO:
derivation -> task -> novelty -> pair -> deterministic validation
-> five-gate Gemma judge -> Luna reviewer -> final dedup -> publish
```

## Getting Started

```bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with the credentials needed for live generation/publication:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

OpenRouter routing defaults to `auto`, preserving provider fallback. Models and providers remain configurable through Make variables.

## Supported Commands

Every dataset exposes the same five command shapes:

```text
<dataset>-smoke
<dataset>-generate
<dataset>-inspect
<dataset>-report
<dataset>-push
```

For example:

```bash
make sft-smoke
make sft-inspect
make sft-report
```

Model suitability can be checked per dataset:

```bash
QUALIFY_MODEL=<openrouter-model-id> make model-qualify-sft
```

Reasoning is disabled whenever a selected model supports disabling it. A model whose reasoning is mandatory/non-disableable is unsuitable for these generation roles.

See `docs/COMMANDS.md` for exact variables and `docs/GENERATION_WORKFLOW.md` for the run sequence.

## Project Structure

```text
configs/       generation configuration
slm_synth/     five dataset packages plus shared runtime
scripts/       repository utilities
docs/          supported workflow documentation
tests/         test suite
Makefile       supported command surface
```

## Testing

```bash
make test
```

## Status

The repository has one supported production path for each of the five dataset products. Legacy generation/orchestration paths have been removed.

## License

MIT. See `LICENSE`.
