# Generation Workflow

Use the same operational sequence for all five dataset products:

1. Qualify the intended model(s) when changing model choices.
2. Run the dataset smoke target.
3. Inspect generated rows/pairs and manifests.
4. Run the dataset report target.
5. Run the production generation target with deliberate sizing variables.
6. Inspect and report the production run.
7. Push only after the final artifact is complete and publish-ready.

The supported Make targets are always:

```text
<dataset>-smoke
<dataset>-generate
<dataset>-inspect
<dataset>-report
<dataset>-push
```

## Pretraining

```bash
make pretrain-smoke
make pretrain-inspect
make pretrain-report

PRETRAIN_TARGET_TOKENS=1000000 \
PRETRAIN_TARGET_RUN=pretrain-production-001 \
make pretrain-generate

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-production-001
make pretrain-report PRETRAIN_REPORT_RUN=pretrain-production-001
make pretrain-push PRETRAIN_REPORT_RUN=pretrain-production-001
```

Production order:

```text
grounded generation
-> deterministic validation
-> Gemma judge
-> Luna reviewer
-> final global dedup
-> accepted-token accounting
-> backfill through the same full path when needed
-> consolidated pretrain.jsonl
```

The five pretrain signals stay internal to this one dataset.

## Generic SFT

Smoke defaults to a tiny `grounded_qa_and_reading` run:

```bash
make sft-smoke
make sft-inspect
make sft-report
```

A production run selects families and controls generated breadth through seeds, derivations per seed, and tasks per derivation:

```bash
SFT_FAMILIES="everyday_conversation programming grounded_qa_and_reading" \
SFT_SEEDS=1 \
SFT_DERIVATIONS_PER_SEED=30 \
SFT_TASKS_PER_DERIVATION=15 \
SFT_GENERATION_RUN=sft-production-001 \
make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-production-001
make sft-report SFT_REPORT_RUN=sft-production-001
make sft-push SFT_PUSH_RUN=sft-production-001
```

Production order:

```text
derivation -> task -> novelty -> answer -> deterministic validation
-> Nemotron judge -> Gemma reviewer -> final dedup
```

## Generic DPO

```bash
make dpo-smoke
make dpo-inspect
make dpo-report
```

Production example:

```bash
DPO_PREFERENCE_DIMENSIONS="helpfulness_and_completeness factual_accuracy instruction_adherence" \
DPO_SEEDS=1 \
DPO_DERIVATIONS_PER_SEED=30 \
DPO_TASKS_PER_DERIVATION=15 \
DPO_GENERATION_RUN=dpo-production-001 \
make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-production-001
make dpo-report DPO_REPORT_RUN=dpo-production-001
make dpo-push DPO_PUSH_RUN=dpo-production-001
```

Production order:

```text
derivation -> task -> novelty -> pair -> deterministic validation
-> Nemotron judge -> Gemma reviewer -> final preference-triple dedup
```

## Distillation SFT

```bash
make distillation-sft-smoke
make distillation-sft-inspect
make distillation-sft-report
```

Production example:

```bash
DISTILLATION_SFT_SIGNALS="arithmetic cloud code debugging" \
DISTILLATION_SFT_SEEDS=1 \
DISTILLATION_SFT_DERIVATIONS_PER_SEED=30 \
DISTILLATION_SFT_TASKS_PER_DERIVATION=15 \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
make distillation-sft-generate

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-production-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-production-001
make distillation-sft-push DISTILLATION_SFT_PUSH_RUN=distillation-sft-production-001
```

Production order:

```text
derivation -> student prompt -> task novelty -> teacher response
-> deterministic validation/response novelty -> Nemotron judge
-> Gemma reviewer -> final prompt/response dedup
```

## Distillation DPO

```bash
make distillation-dpo-smoke
make distillation-dpo-inspect
make distillation-dpo-report
```

Production example:

```bash
DISTILLATION_DPO_DIMENSIONS="factual_accuracy instruction_adherence groundedness" \
DISTILLATION_DPO_SEEDS=1 \
DISTILLATION_DPO_DERIVATIONS_PER_SEED=30 \
DISTILLATION_DPO_TASKS_PER_DERIVATION=15 \
DISTILLATION_DPO_TARGET_RUN=distillation-dpo-production-001 \
make distillation-dpo-generate

make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-production-001
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=distillation-dpo-production-001
make distillation-dpo-push DISTILLATION_DPO_PUSH_RUN=distillation-dpo-production-001
```

Production order:

```text
derivation -> task -> novelty -> pair -> deterministic validation
-> five-gate Gemma judge -> Luna reviewer -> final dedup
```

## Model Qualification

Use dataset-specific qualification targets:

```bash
QUALIFY_MODEL=<model> make model-qualify-pretrain
QUALIFY_MODEL=<model> make model-qualify-sft
QUALIFY_MODEL=<model> make model-qualify-dpo
QUALIFY_MODEL=<model> make model-qualify-distillation-sft
QUALIFY_MODEL=<model> make model-qualify-distillation-dpo
```

Qualification uses the same strict structured-output backend contract as production. Reasoning is disabled whenever supported. Mandatory-reasoning models fail suitability.

## Routing

`OPENROUTER_ROUTING_MODE=auto` is the default and preserves fallback. Use `prefer` with `OPENROUTER_PROVIDER` when a provider should be tried first while preserving fallback. Use `strict` only when provider pinning is intentional.
