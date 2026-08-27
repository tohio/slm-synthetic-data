# ============================================================
# SLM Synthetic Data
# ============================================================

.RECIPEPREFIX := >
MAKEFLAGS += --no-print-directory

PYTHON ?= python

# Shared defaults
MODEL ?= deepseek/deepseek-v4-flash
MAX_TOKENS ?= 4096
OPENROUTER_ROUTING_MODE ?= auto
OPENROUTER_PROVIDER ?=
OPENROUTER_PROVIDER_ORDER ?=
OPENROUTER_PROVIDER_ONLY ?=
OPENROUTER_PROVIDER_IGNORE ?=
OPENROUTER_PROVIDER_SORT ?=
OPENROUTER_PROVIDER_ARG := $(if $(OPENROUTER_PROVIDER),--openrouter-provider $(OPENROUTER_PROVIDER),)
OPENROUTER_ROUTING_ARGS := --openrouter-routing-mode $(OPENROUTER_ROUTING_MODE) $(OPENROUTER_PROVIDER_ARG)
OPENROUTER_ENV := OPENROUTER_ROUTING_MODE="$(OPENROUTER_ROUTING_MODE)" OPENROUTER_PROVIDER="$(OPENROUTER_PROVIDER)" OPENROUTER_PROVIDER_ORDER="$(OPENROUTER_PROVIDER_ORDER)" OPENROUTER_PROVIDER_ONLY="$(OPENROUTER_PROVIDER_ONLY)" OPENROUTER_PROVIDER_IGNORE="$(OPENROUTER_PROVIDER_IGNORE)" OPENROUTER_PROVIDER_SORT="$(OPENROUTER_PROVIDER_SORT)"
QUALIFY_MODEL ?= $(MODEL)
QUALIFY_ROLES ?= all
QUALIFY_MAX_TOKENS ?= 512
QUALIFY_OUTPUT ?= data/model-qualification/$(subst /,_,$(QUALIFY_MODEL)).json
# Pretraining
CONFIG_FILE ?= configs/synthetic.yaml
DATA_DIR ?= data/runs
PROFILE ?= balanced
PRETRAIN_RUN ?= pretrain-smoke-001
PRETRAIN_TARGET_RUN ?= pretrain-target-001
PRETRAIN_REPORT_RUN ?= $(PRETRAIN_RUN)
PRETRAIN_INSPECT_RUN ?= $(PRETRAIN_REPORT_RUN)
PRETRAIN_TOKENS ?= 100000
PRETRAIN_TARGET_TOKENS ?= 1000000
PRETRAIN_BATCH_SIZE ?= 32
PRETRAIN_MAX_TOKENS ?= $(MAX_TOKENS)
PRETRAIN_CONCURRENCY ?= 1
PRETRAIN_TARGET_CONCURRENCY ?= 4
PRETRAIN_MODEL ?= openai/gpt-5.6-luna-pro
PRETRAIN_SIGNAL ?=
PRETRAIN_SIGNAL_ARG := $(if $(PRETRAIN_SIGNAL),--signal $(PRETRAIN_SIGNAL),)
PRETRAIN_JUDGE_MODEL ?= google/gemma-4-31b-it
PRETRAIN_REVIEWER_MODEL ?= openai/gpt-5.6-luna-pro
PRETRAIN_JUDGE_MAX_TOKENS ?= 4096
PRETRAIN_REVIEWER_MAX_TOKENS ?= 4096
PRETRAIN_JUDGE_BATCH_SIZE ?= 10
PRETRAIN_REVIEWER_BATCH_SIZE ?= 10
PRETRAIN_QUALITY_CONCURRENCY ?= 8
PRETRAIN_STAGE_BATCH_ATTEMPTS ?= 3
PRETRAIN_MAX_BACKFILL_ROUNDS ?= 4
PRETRAIN_DIVERSITY_SAMPLE_SIZE ?= 10000
PRETRAIN_DIVERSITY_THRESHOLD ?= 0.80
HF_REPO ?=
HF_NAMESPACE ?= tohio
HF_PRIVATE ?=
HF_PRIVATE_ARG := $(if $(filter true yes 1,$(HF_PRIVATE)),--private,)

# Distillation SFT
DISTILLATION_SFT_RUN ?= distillation-sft-smoke-001
DISTILLATION_SFT_GENERATION_RUN ?= distillation-sft-production-001
DISTILLATION_SFT_REPORT_RUN ?= $(DISTILLATION_SFT_RUN)
DISTILLATION_SFT_INSPECT_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_SIGNALS ?= all
DISTILLATION_SFT_SMOKE_SIGNALS ?= debugging
DISTILLATION_SFT_SMOKE_SIGNALS_EFFECTIVE := $(if $(filter file,$(origin DISTILLATION_SFT_SIGNALS)),$(DISTILLATION_SFT_SMOKE_SIGNALS),$(DISTILLATION_SFT_SIGNALS))
DISTILLATION_SFT_RUN_ROOT ?= data/distillation/runs
DISTILLATION_SFT_SEEDS ?= 1
DISTILLATION_SFT_DERIVATIONS_PER_SEED ?= 30
DISTILLATION_SFT_TASKS_PER_DERIVATION ?= 15
DISTILLATION_SFT_SMOKE_DERIVATIONS_PER_SEED ?= 1
DISTILLATION_SFT_SMOKE_TASKS_PER_DERIVATION ?= 2
DISTILLATION_SFT_ANSWER_BATCH_SIZE ?= 4
DISTILLATION_SFT_JUDGE_BATCH_SIZE ?= 10
DISTILLATION_SFT_REVIEWER_BATCH_SIZE ?= 10
DISTILLATION_SFT_CONCURRENCY ?= 8
DISTILLATION_SFT_CARDINALITY_FILL_ATTEMPTS ?= 3
DISTILLATION_SFT_STAGE_BATCH_ATTEMPTS ?= 3
DISTILLATION_SFT_DERIVATION_MODEL ?= openai/gpt-5.6-luna-pro
DISTILLATION_SFT_TASK_MODEL ?= deepseek/deepseek-v4-flash
DISTILLATION_SFT_ANSWER_MODEL ?= deepseek/deepseek-v4-flash
DISTILLATION_SFT_JUDGE_MODEL ?= nvidia/nemotron-3.5-lightning
DISTILLATION_SFT_REVIEWER_MODEL ?= google/gemma-4-31b-it
DISTILLATION_SFT_DERIVATION_MAX_TOKENS ?= 4096
DISTILLATION_SFT_TASK_MAX_TOKENS ?= 4096
DISTILLATION_SFT_ANSWER_MAX_TOKENS ?= 4096
DISTILLATION_SFT_JUDGE_MAX_TOKENS ?= 4096
DISTILLATION_SFT_REVIEWER_MAX_TOKENS ?= 512
DISTILLATION_SFT_JACCARD_THRESHOLD ?= 0.82
DISTILLATION_SFT_SEQUENCE_THRESHOLD ?= 0.90
DISTILLATION_SFT_DATASET_NAME ?= SLM Synthetic Distillation
DISTILLATION_SFT_PUSH_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(HF_NAMESPACE)/slm-synthetic-distillation-sft)

# Distillation DPO
DISTILLATION_DPO_RUN ?= distillation-dpo-smoke-001
DISTILLATION_DPO_TARGET_RUN ?= distillation-dpo-production-001
DISTILLATION_DPO_REPORT_RUN ?= $(DISTILLATION_DPO_RUN)
DISTILLATION_DPO_INSPECT_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_RUN_ROOT ?= data/distillation-dpo/runs
DISTILLATION_DPO_DIMENSIONS ?= all
DISTILLATION_DPO_SMOKE_DIMENSIONS ?= factual_accuracy
DISTILLATION_DPO_SMOKE_DIMENSIONS_EFFECTIVE := $(if $(filter command line,$(origin DISTILLATION_DPO_DIMENSIONS)),$(DISTILLATION_DPO_DIMENSIONS),$(DISTILLATION_DPO_SMOKE_DIMENSIONS))
DISTILLATION_DPO_SEEDS ?= 1
DISTILLATION_DPO_DERIVATIONS_PER_SEED ?= 30
DISTILLATION_DPO_TASKS_PER_DERIVATION ?= 15
DISTILLATION_DPO_SMOKE_DERIVATIONS_PER_SEED ?= 1
DISTILLATION_DPO_SMOKE_TASKS_PER_DERIVATION ?= 2
DISTILLATION_DPO_PAIR_BATCH_SIZE ?= 4
DISTILLATION_DPO_JUDGE_BATCH_SIZE ?= 10
DISTILLATION_DPO_REVIEWER_BATCH_SIZE ?= 10
DISTILLATION_DPO_CONCURRENCY ?= 8
DISTILLATION_DPO_CARDINALITY_FILL_ATTEMPTS ?= 3
DISTILLATION_DPO_STAGE_BATCH_ATTEMPTS ?= 3
DISTILLATION_DPO_DERIVATION_MODEL ?= openai/gpt-5.6-luna-pro
DISTILLATION_DPO_TASK_MODEL ?= deepseek/deepseek-v4-flash
DISTILLATION_DPO_PAIR_MODEL ?= deepseek/deepseek-v4-flash
DISTILLATION_DPO_JUDGE_MODEL ?= google/gemma-4-31b-it
DISTILLATION_DPO_REVIEWER_MODEL ?= openai/gpt-5.6-luna-pro
DISTILLATION_DPO_DERIVATION_MAX_TOKENS ?= 4096
DISTILLATION_DPO_TASK_MAX_TOKENS ?= 4096
DISTILLATION_DPO_PAIR_MAX_TOKENS ?= 4096
DISTILLATION_DPO_JUDGE_MAX_TOKENS ?= 4096
DISTILLATION_DPO_REVIEWER_MAX_TOKENS ?= 512
DISTILLATION_DPO_JACCARD_THRESHOLD ?= 0.82
DISTILLATION_DPO_SEQUENCE_THRESHOLD ?= 0.90
DISTILLATION_DPO_DATASET_NAME ?= SLM Synthetic Distillation DPO
DISTILLATION_DPO_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
DISTILLATION_DPO_PUSH_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_HF_REPO ?= $(HF_NAMESPACE)/slm-synthetic-distillation-dpo

# SFT
SFT_RUN ?= sft-smoke-001
SFT_GENERATION_RUN ?= sft-production-001
SFT_REPORT_RUN ?= $(SFT_RUN)
SFT_INSPECT_RUN ?= $(SFT_REPORT_RUN)
SFT_FAMILIES ?= all
SFT_SMOKE_FAMILIES ?= grounded_qa_and_reading
SFT_RUN_ROOT ?= data/sft/runs
SFT_SEEDS ?= 1
SFT_DERIVATIONS_PER_SEED ?= 30
SFT_TASKS_PER_DERIVATION ?= 15
SFT_SMOKE_DERIVATIONS_PER_SEED ?= 1
SFT_SMOKE_TASKS_PER_DERIVATION ?= 2
SFT_ANSWER_BATCH_SIZE ?= 4
SFT_JUDGE_BATCH_SIZE ?= 10
SFT_REVIEWER_BATCH_SIZE ?= 10
SFT_CONCURRENCY ?= 8
SFT_CARDINALITY_FILL_ATTEMPTS ?= 3
SFT_STAGE_BATCH_ATTEMPTS ?= 3
SFT_DERIVATION_MODEL ?= openai/gpt-5.6-luna-pro
SFT_TASK_MODEL ?= deepseek/deepseek-v4-flash
SFT_ANSWER_MODEL ?= deepseek/deepseek-v4-flash
SFT_JUDGE_MODEL ?= nvidia/nemotron-3.5-lightning
SFT_REVIEWER_MODEL ?= google/gemma-4-31b-it
SFT_DERIVATION_MAX_TOKENS ?= 4096
SFT_TASK_MAX_TOKENS ?= 4096
SFT_ANSWER_MAX_TOKENS ?= 4096
SFT_JUDGE_MAX_TOKENS ?= 4096
SFT_REVIEWER_MAX_TOKENS ?= 512
SFT_JACCARD_THRESHOLD ?= 0.82
SFT_SEQUENCE_THRESHOLD ?= 0.90
SFT_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
SFT_PUSH_RUN ?= $(SFT_REPORT_RUN)
SFT_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(if $(HF_NAMESPACE),$(HF_NAMESPACE)/slm-synthetic-sft,))
SFT_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter file,$(origin SFT_FAMILIES)),$(SFT_SMOKE_FAMILIES),$(SFT_FAMILIES))

# DPO
DPO_RUN ?= dpo-smoke-001
DPO_GENERATION_RUN ?= dpo-candidate-001
DPO_REPORT_RUN ?= $(DPO_RUN)
DPO_INSPECT_RUN ?= $(DPO_REPORT_RUN)
DPO_PREFERENCE_DIMENSIONS ?= all
DPO_SMOKE_PREFERENCE_DIMENSIONS ?= instruction_adherence
DPO_SEEDS ?= 1
DPO_SMOKE_DERIVATIONS_PER_SEED ?= 2
DPO_SMOKE_TASKS_PER_DERIVATION ?= 2
DPO_DERIVATIONS_PER_SEED ?= 30
DPO_TASKS_PER_DERIVATION ?= 15
DPO_PAIR_BATCH_SIZE ?= 4
DPO_JUDGE_BATCH_SIZE ?= 10
DPO_REVIEWER_BATCH_SIZE ?= 10
DPO_CONCURRENCY ?= 8
DPO_CARDINALITY_FILL_ATTEMPTS ?= 3
DPO_STAGE_BATCH_ATTEMPTS ?= 3
DPO_RUN_ROOT ?= data/dpo/runs
DPO_DERIVATION_MODEL ?= openai/gpt-5.6-luna-pro
DPO_TASK_MODEL ?= deepseek/deepseek-v4-flash
DPO_PAIR_MODEL ?= deepseek/deepseek-v4-flash
DPO_JUDGE_MODEL ?= nvidia/nemotron-3.5-lightning
DPO_REVIEWER_MODEL ?= google/gemma-4-31b-it
DPO_DERIVATION_MAX_TOKENS ?= 4096
DPO_TASK_MAX_TOKENS ?= 4096
DPO_PAIR_MAX_TOKENS ?= 4096
DPO_JUDGE_MAX_TOKENS ?= 4096
DPO_REVIEWER_MAX_TOKENS ?= 512
DPO_JACCARD_THRESHOLD ?= 0.82
DPO_SEQUENCE_THRESHOLD ?= 0.90
DPO_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
DPO_PUSH_RUN ?= $(DPO_REPORT_RUN)
DPO_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(if $(HF_NAMESPACE),$(HF_NAMESPACE)/slm-synthetic-dpo,))
DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE := $(if $(filter file,$(origin DPO_PREFERENCE_DIMENSIONS)),$(DPO_SMOKE_PREFERENCE_DIMENSIONS),$(DPO_PREFERENCE_DIMENSIONS))

.PHONY: help \
	pretrain-smoke pretrain-generate pretrain-inspect pretrain-report pretrain-push \
	sft-smoke sft-generate sft-inspect sft-report sft-push \
	dpo-smoke dpo-generate dpo-inspect dpo-report dpo-push \
	distillation-sft-smoke distillation-sft-generate distillation-sft-inspect distillation-sft-report distillation-sft-push \
	distillation-dpo-smoke distillation-dpo-generate distillation-dpo-inspect distillation-dpo-report distillation-dpo-push \
	model-qualify model-qualify-pretrain model-qualify-sft model-qualify-dpo \
	model-qualify-distillation-sft model-qualify-distillation-dpo model-qualify-all \
	test clean

help:
> @echo ""
> @echo "SLM Synthetic Data"
> @echo "=================="
> @echo ""
> @echo "Dataset                 Smoke                         Generate                         Inspect                         Report                         Push"
> @echo "pretrain                pretrain-smoke                pretrain-generate                pretrain-inspect                pretrain-report                pretrain-push"
> @echo "sft                     sft-smoke                     sft-generate                     sft-inspect                     sft-report                     sft-push"
> @echo "dpo                     dpo-smoke                     dpo-generate                     dpo-inspect                     dpo-report                     dpo-push"
> @echo "distillation-sft        distillation-sft-smoke        distillation-sft-generate        distillation-sft-inspect        distillation-sft-report        distillation-sft-push"
> @echo "distillation-dpo        distillation-dpo-smoke        distillation-dpo-generate        distillation-dpo-inspect        distillation-dpo-report        distillation-dpo-push"
> @echo ""
> @echo "Qualification:"
> @echo "  model-qualify"
> @echo "  model-qualify-pretrain"
> @echo "  model-qualify-sft"
> @echo "  model-qualify-dpo"
> @echo "  model-qualify-distillation-sft"
> @echo "  model-qualify-distillation-dpo"
> @echo "  model-qualify-all"
> @echo ""
> @echo "Maintenance:"
> @echo "  test"
> @echo "  clean"

pretrain-smoke:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(PRETRAIN_TOKENS) \
>   --batch-size $(PRETRAIN_BATCH_SIZE) \
>   --model $(PRETRAIN_MODEL) \
>   --concurrency $(PRETRAIN_CONCURRENCY) \
>   --max-tokens $(PRETRAIN_MAX_TOKENS) \
>   --run $(PRETRAIN_RUN) \
>   $(if $(HF_REPO),--hf_repo $(HF_REPO),)
> $(PYTHON) -m slm_synth.pretrain.preflight_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.pretrain.pipeline \
>   --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG) \
>   --judge-model $(PRETRAIN_JUDGE_MODEL) \
>   --reviewer-model $(PRETRAIN_REVIEWER_MODEL) \
>   --judge-max-tokens $(PRETRAIN_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(PRETRAIN_REVIEWER_MAX_TOKENS) \
>   --judge-batch-size $(PRETRAIN_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(PRETRAIN_REVIEWER_BATCH_SIZE) \
>   --quality-concurrency $(PRETRAIN_QUALITY_CONCURRENCY) \
>   --stage-batch-attempts $(PRETRAIN_STAGE_BATCH_ATTEMPTS) \
>   --max-backfill-rounds $(PRETRAIN_MAX_BACKFILL_ROUNDS) \
>   $(OPENROUTER_ROUTING_ARGS)
> $(PYTHON) -m slm_synth.pretrain.report_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(MAKE) pretrain-report PRETRAIN_REPORT_RUN=$(PRETRAIN_RUN)

pretrain-generate:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(PRETRAIN_TARGET_TOKENS) \
>   --batch-size $(PRETRAIN_BATCH_SIZE) \
>   --model $(PRETRAIN_MODEL) \
>   --concurrency $(PRETRAIN_TARGET_CONCURRENCY) \
>   --max-tokens $(PRETRAIN_MAX_TOKENS) \
>   --run $(PRETRAIN_TARGET_RUN) \
>   $(if $(HF_REPO),--hf_repo $(HF_REPO),)
> $(PYTHON) -m slm_synth.pretrain.preflight_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.pretrain.pipeline \
>   --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG) \
>   --judge-model $(PRETRAIN_JUDGE_MODEL) \
>   --reviewer-model $(PRETRAIN_REVIEWER_MODEL) \
>   --judge-max-tokens $(PRETRAIN_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(PRETRAIN_REVIEWER_MAX_TOKENS) \
>   --judge-batch-size $(PRETRAIN_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(PRETRAIN_REVIEWER_BATCH_SIZE) \
>   --quality-concurrency $(PRETRAIN_QUALITY_CONCURRENCY) \
>   --stage-batch-attempts $(PRETRAIN_STAGE_BATCH_ATTEMPTS) \
>   --max-backfill-rounds $(PRETRAIN_MAX_BACKFILL_ROUNDS) \
>   $(OPENROUTER_ROUTING_ARGS)
> $(PYTHON) -m slm_synth.pretrain.report_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(MAKE) pretrain-report PRETRAIN_REPORT_RUN=$(PRETRAIN_TARGET_RUN)

pretrain-report:
> $(PYTHON) -m slm_synth.pretrain.pipeline --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG) --verify-only
> $(PYTHON) -m slm_synth.pretrain.manifest \
>   --config $(CONFIG_FILE) \
>   --generation-run $(PRETRAIN_REPORT_RUN)
> $(PYTHON) -m slm_synth.pretrain.report_diversity \
>   --config $(CONFIG_FILE) \
>   --stage deduped \
>   --sample-size $(PRETRAIN_DIVERSITY_SAMPLE_SIZE) \
>   --near-duplicate-threshold $(PRETRAIN_DIVERSITY_THRESHOLD) \
>   --require-clean
> @test -z "$(PRETRAIN_REPORT_RUN)" || $(PYTHON) -m slm_synth.cards build --kind pretrain --run-dir data/runs/$(PRETRAIN_REPORT_RUN)
> @if [ -n "$(PRETRAIN_REPORT_RUN)" ]; then $(PYTHON) -m slm_synth.manifest_totals normalize --kind pretrain --run-dir data/runs/$(PRETRAIN_REPORT_RUN); fi

pretrain-inspect:
> @echo "== pretraining files =="
> @find $(DATA_DIR)/$(PRETRAIN_INSPECT_RUN) -type f 2>/dev/null | sort | tail -n 50
> @echo "== pretraining sample rows =="
> @test ! -f $(DATA_DIR)/$(PRETRAIN_INSPECT_RUN)/deduped/pretrain.jsonl || (echo "--- consolidated pretrain.jsonl"; head -n 3 $(DATA_DIR)/$(PRETRAIN_INSPECT_RUN)/deduped/pretrain.jsonl)

pretrain-push:
> $(PYTHON) -m slm_synth.pretrain.push_hf --config $(CONFIG_FILE) $(if $(HF_REPO),--repo-id $(HF_REPO),)

distillation-sft-smoke:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_sft.pipeline \
>   --signals $(DISTILLATION_SFT_SMOKE_SIGNALS_EFFECTIVE) \
>   --generation-run $(DISTILLATION_SFT_RUN) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_RUN) \
>   --seeds $(DISTILLATION_SFT_SEEDS) \
>   --derivations-per-seed $(DISTILLATION_SFT_SMOKE_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DISTILLATION_SFT_SMOKE_TASKS_PER_DERIVATION) \
>   --answer-batch-size $(DISTILLATION_SFT_ANSWER_BATCH_SIZE) \
>   --judge-batch-size $(DISTILLATION_SFT_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DISTILLATION_SFT_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_CONCURRENCY) \
>   --cardinality-fill-attempts $(DISTILLATION_SFT_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DISTILLATION_SFT_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DISTILLATION_SFT_DERIVATION_MODEL) \
>   --task-model $(DISTILLATION_SFT_TASK_MODEL) \
>   --answer-model $(DISTILLATION_SFT_ANSWER_MODEL) \
>   --judge-model $(DISTILLATION_SFT_JUDGE_MODEL) \
>   --reviewer-model $(DISTILLATION_SFT_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DISTILLATION_SFT_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DISTILLATION_SFT_TASK_MAX_TOKENS) \
>   --answer-max-tokens $(DISTILLATION_SFT_ANSWER_MAX_TOKENS) \
>   --judge-max-tokens $(DISTILLATION_SFT_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DISTILLATION_SFT_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(DISTILLATION_SFT_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DISTILLATION_SFT_SEQUENCE_THRESHOLD)
> $(MAKE) distillation-sft-report DISTILLATION_SFT_REPORT_RUN=$(DISTILLATION_SFT_RUN)

distillation-sft-generate:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_sft.pipeline \
>   --signals $(DISTILLATION_SFT_SIGNALS) \
>   --generation-run $(DISTILLATION_SFT_GENERATION_RUN) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_GENERATION_RUN) \
>   --seeds $(DISTILLATION_SFT_SEEDS) \
>   --derivations-per-seed $(DISTILLATION_SFT_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DISTILLATION_SFT_TASKS_PER_DERIVATION) \
>   --answer-batch-size $(DISTILLATION_SFT_ANSWER_BATCH_SIZE) \
>   --judge-batch-size $(DISTILLATION_SFT_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DISTILLATION_SFT_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_CONCURRENCY) \
>   --cardinality-fill-attempts $(DISTILLATION_SFT_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DISTILLATION_SFT_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DISTILLATION_SFT_DERIVATION_MODEL) \
>   --task-model $(DISTILLATION_SFT_TASK_MODEL) \
>   --answer-model $(DISTILLATION_SFT_ANSWER_MODEL) \
>   --judge-model $(DISTILLATION_SFT_JUDGE_MODEL) \
>   --reviewer-model $(DISTILLATION_SFT_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DISTILLATION_SFT_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DISTILLATION_SFT_TASK_MAX_TOKENS) \
>   --answer-max-tokens $(DISTILLATION_SFT_ANSWER_MAX_TOKENS) \
>   --judge-max-tokens $(DISTILLATION_SFT_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DISTILLATION_SFT_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(DISTILLATION_SFT_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DISTILLATION_SFT_SEQUENCE_THRESHOLD)
> $(MAKE) distillation-sft-report DISTILLATION_SFT_REPORT_RUN=$(DISTILLATION_SFT_GENERATION_RUN)

distillation-sft-report:
> $(PYTHON) -m slm_synth.distillation_sft.cli report-coverage \
>   --run-manifest $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/manifests/$(DISTILLATION_SFT_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.distillation_sft.cli build-dataset-card \
>   --run-manifest $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/manifests/$(DISTILLATION_SFT_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/README.md \
>   --dataset-name "$(DISTILLATION_SFT_DATASET_NAME)"
> $(PYTHON) -m slm_synth.cards build --kind distillation-sft --run-dir data/distillation/runs/$(DISTILLATION_SFT_REPORT_RUN)
> $(PYTHON) -m slm_synth.manifest_totals normalize --kind distillation-sft --run-dir data/distillation/runs/$(DISTILLATION_SFT_REPORT_RUN)

distillation-sft-inspect:
> @echo "== distillation files =="
> @find $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== distillation sample rows =="
> @find $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

distillation-sft-push:
> test -n "$(DISTILLATION_SFT_HF_REPO)" || (echo "DISTILLATION_SFT_HF_REPO or HF_REPO is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.distillation_sft.push_hf \
>   --dataset-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN)/datasets \
>   --run-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN) \
>   --repo-id $(DISTILLATION_SFT_HF_REPO) $(HF_PRIVATE_ARG)

distillation-dpo-smoke:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_dpo.pipeline \
>   --dimensions $(DISTILLATION_DPO_SMOKE_DIMENSIONS_EFFECTIVE) \
>   --generation-run $(DISTILLATION_DPO_RUN) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_RUN) \
>   --seeds $(DISTILLATION_DPO_SEEDS) \
>   --derivations-per-seed $(DISTILLATION_DPO_SMOKE_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DISTILLATION_DPO_SMOKE_TASKS_PER_DERIVATION) \
>   --pair-batch-size $(DISTILLATION_DPO_PAIR_BATCH_SIZE) \
>   --judge-batch-size $(DISTILLATION_DPO_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DISTILLATION_DPO_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_DPO_CONCURRENCY) \
>   --cardinality-fill-attempts $(DISTILLATION_DPO_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DISTILLATION_DPO_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DISTILLATION_DPO_DERIVATION_MODEL) \
>   --task-model $(DISTILLATION_DPO_TASK_MODEL) \
>   --pair-model $(DISTILLATION_DPO_PAIR_MODEL) \
>   --judge-model $(DISTILLATION_DPO_JUDGE_MODEL) \
>   --reviewer-model $(DISTILLATION_DPO_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DISTILLATION_DPO_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DISTILLATION_DPO_TASK_MAX_TOKENS) \
>   --pair-max-tokens $(DISTILLATION_DPO_PAIR_MAX_TOKENS) \
>   --judge-max-tokens $(DISTILLATION_DPO_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DISTILLATION_DPO_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(DISTILLATION_DPO_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DISTILLATION_DPO_SEQUENCE_THRESHOLD)
> $(MAKE) distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=$(DISTILLATION_DPO_RUN)

distillation-dpo-generate:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_dpo.pipeline \
>   --dimensions $(DISTILLATION_DPO_DIMENSIONS) \
>   --generation-run $(DISTILLATION_DPO_TARGET_RUN) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_TARGET_RUN) \
>   --seeds $(DISTILLATION_DPO_SEEDS) \
>   --derivations-per-seed $(DISTILLATION_DPO_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DISTILLATION_DPO_TASKS_PER_DERIVATION) \
>   --pair-batch-size $(DISTILLATION_DPO_PAIR_BATCH_SIZE) \
>   --judge-batch-size $(DISTILLATION_DPO_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DISTILLATION_DPO_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_DPO_CONCURRENCY) \
>   --cardinality-fill-attempts $(DISTILLATION_DPO_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DISTILLATION_DPO_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DISTILLATION_DPO_DERIVATION_MODEL) \
>   --task-model $(DISTILLATION_DPO_TASK_MODEL) \
>   --pair-model $(DISTILLATION_DPO_PAIR_MODEL) \
>   --judge-model $(DISTILLATION_DPO_JUDGE_MODEL) \
>   --reviewer-model $(DISTILLATION_DPO_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DISTILLATION_DPO_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DISTILLATION_DPO_TASK_MAX_TOKENS) \
>   --pair-max-tokens $(DISTILLATION_DPO_PAIR_MAX_TOKENS) \
>   --judge-max-tokens $(DISTILLATION_DPO_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DISTILLATION_DPO_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(DISTILLATION_DPO_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DISTILLATION_DPO_SEQUENCE_THRESHOLD)
> $(MAKE) distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=$(DISTILLATION_DPO_TARGET_RUN)

distillation-dpo-report:
> $(PYTHON) -m slm_synth.distillation_dpo.cli report-coverage \
>   --input $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/datasets \
>   --holdout-registry $(DISTILLATION_DPO_HOLDOUT_REGISTRY) \
>   --output $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.distillation_dpo.cli build-dataset-card \
>   --run-manifest $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/manifests/$(DISTILLATION_DPO_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/README.md \
>   --dataset-name "$(DISTILLATION_DPO_DATASET_NAME)"
> $(PYTHON) -m slm_synth.cards build --kind distillation-dpo --run-dir data/distillation-dpo/runs/$(DISTILLATION_DPO_REPORT_RUN)
> $(PYTHON) -m slm_synth.manifest_totals normalize --kind distillation-dpo --run-dir data/distillation-dpo/runs/$(DISTILLATION_DPO_REPORT_RUN)

distillation-dpo-inspect:
> @echo "== distillation DPO files =="
> @find $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== distillation DPO sample rows =="
> @find $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

distillation-dpo-push:
> $(PYTHON) -m slm_synth.distillation_dpo.push_hf \
>   --dataset-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_PUSH_RUN)/datasets \
>   --run-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_PUSH_RUN) \
>   --repo-id $(DISTILLATION_DPO_HF_REPO) $(HF_PRIVATE_ARG)




sft-smoke:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.sft.pipeline \
>   --families $(SFT_SMOKE_FAMILIES_EFFECTIVE) \
>   --generation-run $(SFT_RUN) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_RUN) \
>   --seeds $(SFT_SEEDS) \
>   --derivations-per-seed $(SFT_SMOKE_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(SFT_SMOKE_TASKS_PER_DERIVATION) \
>   --answer-batch-size $(SFT_ANSWER_BATCH_SIZE) \
>   --judge-batch-size $(SFT_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(SFT_REVIEWER_BATCH_SIZE) \
>   --concurrency $(SFT_CONCURRENCY) \
>   --cardinality-fill-attempts $(SFT_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(SFT_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(SFT_DERIVATION_MODEL) \
>   --task-model $(SFT_TASK_MODEL) \
>   --answer-model $(SFT_ANSWER_MODEL) \
>   --judge-model $(SFT_JUDGE_MODEL) \
>   --reviewer-model $(SFT_REVIEWER_MODEL) \
>   --derivation-max-tokens $(SFT_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(SFT_TASK_MAX_TOKENS) \
>   --answer-max-tokens $(SFT_ANSWER_MAX_TOKENS) \
>   --judge-max-tokens $(SFT_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(SFT_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(SFT_JACCARD_THRESHOLD) \
>   --sequence-threshold $(SFT_SEQUENCE_THRESHOLD)
> $(MAKE) sft-report SFT_REPORT_RUN=$(SFT_RUN)

sft-generate:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.sft.pipeline \
>   --families $(SFT_FAMILIES) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_GENERATION_RUN) \
>   --seeds $(SFT_SEEDS) \
>   --derivations-per-seed $(SFT_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(SFT_TASKS_PER_DERIVATION) \
>   --answer-batch-size $(SFT_ANSWER_BATCH_SIZE) \
>   --judge-batch-size $(SFT_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(SFT_REVIEWER_BATCH_SIZE) \
>   --concurrency $(SFT_CONCURRENCY) \
>   --cardinality-fill-attempts $(SFT_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(SFT_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(SFT_DERIVATION_MODEL) \
>   --task-model $(SFT_TASK_MODEL) \
>   --answer-model $(SFT_ANSWER_MODEL) \
>   --judge-model $(SFT_JUDGE_MODEL) \
>   --reviewer-model $(SFT_REVIEWER_MODEL) \
>   --derivation-max-tokens $(SFT_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(SFT_TASK_MAX_TOKENS) \
>   --answer-max-tokens $(SFT_ANSWER_MAX_TOKENS) \
>   --judge-max-tokens $(SFT_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(SFT_REVIEWER_MAX_TOKENS) \
>   --routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --jaccard-threshold $(SFT_JACCARD_THRESHOLD) \
>   --sequence-threshold $(SFT_SEQUENCE_THRESHOLD)
> $(MAKE) sft-report SFT_REPORT_RUN=$(SFT_GENERATION_RUN)

sft-report:
> $(PYTHON) -m slm_synth.sft.cli report-coverage \
>   --input $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/datasets \
>   --run-manifest $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/manifests/$(SFT_REPORT_RUN).manifest.json \
>   --holdout-registry $(SFT_HOLDOUT_REGISTRY) \
>   --output $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.cards build --kind sft --run-dir data/sft/runs/$(SFT_REPORT_RUN)
> $(PYTHON) -m slm_synth.manifest_totals normalize --kind sft --run-dir data/sft/runs/$(SFT_REPORT_RUN)

sft-inspect:
> @echo "== SFT files =="
> @find $(SFT_RUN_ROOT)/$(SFT_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== SFT sample rows =="
> @find $(SFT_RUN_ROOT)/$(SFT_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

sft-push:
> test -n "$(SFT_HF_REPO)" || (echo "SFT_HF_REPO or HF_REPO is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.sft.push_hf \
>   --dataset-dir $(SFT_RUN_ROOT)/$(SFT_PUSH_RUN)/datasets \
>   --run-dir $(SFT_RUN_ROOT)/$(SFT_PUSH_RUN) \
>   --repo-id $(SFT_HF_REPO) $(HF_PRIVATE_ARG)

dpo-smoke:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.dpo.pipeline \
>   --dimensions $(DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE) \
>   --generation-run $(DPO_RUN) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_RUN) \
>   --seeds $(DPO_SEEDS) \
>   --derivations-per-seed $(DPO_SMOKE_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DPO_SMOKE_TASKS_PER_DERIVATION) \
>   --pair-batch-size $(DPO_PAIR_BATCH_SIZE) \
>   --judge-batch-size $(DPO_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DPO_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DPO_CONCURRENCY) \
>   --cardinality-fill-attempts $(DPO_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DPO_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DPO_DERIVATION_MODEL) \
>   --task-model $(DPO_TASK_MODEL) \
>   --pair-model $(DPO_PAIR_MODEL) \
>   --judge-model $(DPO_JUDGE_MODEL) \
>   --reviewer-model $(DPO_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DPO_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DPO_TASK_MAX_TOKENS) \
>   --pair-max-tokens $(DPO_PAIR_MAX_TOKENS) \
>   --judge-max-tokens $(DPO_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DPO_REVIEWER_MAX_TOKENS) \
>   --jaccard-threshold $(DPO_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DPO_SEQUENCE_THRESHOLD) \
>   $(OPENROUTER_ROUTING_ARGS)
> $(MAKE) dpo-report DPO_REPORT_RUN=$(DPO_RUN)

dpo-generate:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.dpo.pipeline \
>   --dimensions $(DPO_PREFERENCE_DIMENSIONS) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_GENERATION_RUN) \
>   --seeds $(DPO_SEEDS) \
>   --derivations-per-seed $(DPO_DERIVATIONS_PER_SEED) \
>   --tasks-per-derivation $(DPO_TASKS_PER_DERIVATION) \
>   --pair-batch-size $(DPO_PAIR_BATCH_SIZE) \
>   --judge-batch-size $(DPO_JUDGE_BATCH_SIZE) \
>   --reviewer-batch-size $(DPO_REVIEWER_BATCH_SIZE) \
>   --concurrency $(DPO_CONCURRENCY) \
>   --cardinality-fill-attempts $(DPO_CARDINALITY_FILL_ATTEMPTS) \
>   --stage-batch-attempts $(DPO_STAGE_BATCH_ATTEMPTS) \
>   --derivation-model $(DPO_DERIVATION_MODEL) \
>   --task-model $(DPO_TASK_MODEL) \
>   --pair-model $(DPO_PAIR_MODEL) \
>   --judge-model $(DPO_JUDGE_MODEL) \
>   --reviewer-model $(DPO_REVIEWER_MODEL) \
>   --derivation-max-tokens $(DPO_DERIVATION_MAX_TOKENS) \
>   --task-max-tokens $(DPO_TASK_MAX_TOKENS) \
>   --pair-max-tokens $(DPO_PAIR_MAX_TOKENS) \
>   --judge-max-tokens $(DPO_JUDGE_MAX_TOKENS) \
>   --reviewer-max-tokens $(DPO_REVIEWER_MAX_TOKENS) \
>   --jaccard-threshold $(DPO_JACCARD_THRESHOLD) \
>   --sequence-threshold $(DPO_SEQUENCE_THRESHOLD) \
>   $(OPENROUTER_ROUTING_ARGS)
> $(MAKE) dpo-report DPO_REPORT_RUN=$(DPO_GENERATION_RUN)

dpo-report:
> $(PYTHON) -m slm_synth.dpo.cli report-coverage \
>   --input $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/datasets \
>   --holdout-registry $(DPO_HOLDOUT_REGISTRY) \
>   --run-manifest $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/manifests/$(DPO_REPORT_RUN).manifest.json \
>   --output $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.cards build --kind dpo --run-dir $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)
> $(PYTHON) -m slm_synth.manifest_totals normalize --kind dpo --run-dir $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)

dpo-inspect:
> @echo "== DPO files =="
> @find $(DPO_RUN_ROOT)/$(DPO_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== DPO sample rows =="
> @find $(DPO_RUN_ROOT)/$(DPO_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

dpo-push:
> test -n "$(DPO_HF_REPO)" || (echo "DPO_HF_REPO or HF_REPO is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.dpo.push_hf \
>   --dataset-dir $(DPO_RUN_ROOT)/$(DPO_PUSH_RUN)/datasets \
>   --run-dir $(DPO_RUN_ROOT)/$(DPO_PUSH_RUN) \
>   --repo-id $(DPO_HF_REPO) $(HF_PRIVATE_ARG)

model-qualify:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.qualify_model \
>   --model $(QUALIFY_MODEL) \
>   --roles $(QUALIFY_ROLES) \
>   --max-tokens $(QUALIFY_MAX_TOKENS) \
>   --openrouter-routing-mode $(OPENROUTER_ROUTING_MODE) \
>   $(OPENROUTER_PROVIDER_ARG) \
>   --output $(QUALIFY_OUTPUT)

model-qualify-pretrain:
> $(MAKE) model-qualify QUALIFY_ROLES=pretrain

model-qualify-sft:
> $(MAKE) model-qualify QUALIFY_ROLES=sft

model-qualify-dpo:
> $(MAKE) model-qualify QUALIFY_ROLES=dpo

model-qualify-distillation-sft:
> $(MAKE) model-qualify QUALIFY_ROLES=distillation-sft

model-qualify-distillation-dpo:
> $(MAKE) model-qualify QUALIFY_ROLES=distillation-dpo

model-qualify-all:
> $(MAKE) model-qualify QUALIFY_ROLES=all

test:
> $(PYTHON) -m compileall -q slm_synth tests
> pytest -q

clean:
> rm -rf $(DATA_DIR) data/distillation data/sft data/dpo
