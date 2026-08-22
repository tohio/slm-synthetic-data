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
COST_GENERATOR_MODEL ?= $(MODEL)
COST_JUDGE_MODEL ?= $(MODEL)
COST_REVIEWER_MODEL ?= $(MODEL)
COST_CANDIDATES ?= 1000
COST_TARGET_ACCEPTED ?=
COST_TARGET_TOKENS ?=
COST_AVERAGE_ACCEPTED_TOKENS ?= 500
COST_TARGET_ACCEPTED_ARG := $(if $(COST_TARGET_ACCEPTED),--target-accepted $(COST_TARGET_ACCEPTED),)
COST_TARGET_TOKENS_ARG := $(if $(COST_TARGET_TOKENS),--target-tokens $(COST_TARGET_TOKENS),)
COST_OUTPUT ?=
COST_OUTPUT_ARG := $(if $(COST_OUTPUT),--output $(COST_OUTPUT),)

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
PRETRAIN_MODEL ?= $(MODEL)
PRETRAIN_SIGNAL ?=
PRETRAIN_SIGNAL_ARG := $(if $(PRETRAIN_SIGNAL),--signal $(PRETRAIN_SIGNAL),)
PRETRAIN_STAGE ?= deduped
PRETRAIN_DIVERSITY_SAMPLE_SIZE ?= 10000
PRETRAIN_DIVERSITY_THRESHOLD ?= 0.80
HF_REPO ?=
HF_NAMESPACE ?= tohio
HF_PRIVATE ?=
HF_PRIVATE_ARG := $(if $(filter true yes 1,$(HF_PRIVATE)),--private,)

# Distillation SFT
DISTILLATION_SFT_RUN ?= distillation-sft-smoke-001
DISTILLATION_SFT_GENERATION_RUN ?= distillation-sft-candidate-001
DISTILLATION_SFT_REPORT_RUN ?= $(DISTILLATION_SFT_RUN)
DISTILLATION_SFT_INSPECT_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_ADJUDICATION_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_ADJUDICATIONS ?=
DISTILLATION_SFT_CANDIDATE_COUNTS ?=
DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL ?= 2
DISTILLATION_SFT_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)
DISTILLATION_SFT_CONCURRENCY ?= $(PRETRAIN_CONCURRENCY)
DISTILLATION_SFT_GENERATION_CONCURRENCY ?= $(PRETRAIN_TARGET_CONCURRENCY)
DISTILLATION_SFT_SIGNALS ?=
DISTILLATION_SFT_SIGNALS_ARG := $(if $(filter all,$(DISTILLATION_SFT_SIGNALS)),,$(if $(DISTILLATION_SFT_SIGNALS),--signals $(DISTILLATION_SFT_SIGNALS),))
DISTILLATION_SFT_INITIAL_CONCURRENCY ?= 8
DISTILLATION_SFT_INITIAL_BATCH_SIZE ?= 4
DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES ?= 4
DISTILLATION_SFT_RUN_ROOT ?= data/distillation/runs
DISTILLATION_SFT_MODEL ?= $(MODEL)
DISTILLATION_SFT_MAX_TOKENS ?= 4096
DISTILLATION_SFT_DATASET_NAME ?= SLM Synthetic Distillation
DISTILLATION_SFT_PUSH_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(HF_NAMESPACE)/slm-synthetic-distillation-sft)

# Distillation DPO
DISTILLATION_DPO_RUN ?= distillation-dpo-smoke-001
DISTILLATION_DPO_TARGET_RUN ?= distillation-dpo-target-001
DISTILLATION_DPO_REPORT_RUN ?= $(DISTILLATION_DPO_RUN)
DISTILLATION_DPO_INSPECT_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_FAMILIES ?= all
DISTILLATION_DPO_SMOKE_FAMILIES ?= teacher_response_preference
DISTILLATION_DPO_TARGET_PAIRS ?= 15000
DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY ?= 1000
DISTILLATION_DPO_RUN_ROOT ?= data/distillation-dpo/runs
DISTILLATION_DPO_MODEL ?= $(MODEL)
DISTILLATION_DPO_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)
DISTILLATION_DPO_CONCURRENCY ?= $(PRETRAIN_CONCURRENCY)
DISTILLATION_DPO_TARGET_CONCURRENCY ?= $(PRETRAIN_TARGET_CONCURRENCY)
DISTILLATION_DPO_INITIAL_CONCURRENCY ?= 8
DISTILLATION_DPO_INITIAL_BATCH_SIZE ?= 4
DISTILLATION_DPO_BATCH_INCREASE_SUCCESSES ?= 4
DISTILLATION_DPO_MAX_TOKENS ?= 4096
DISTILLATION_DPO_DATASET_NAME ?= SLM Synthetic Distillation DPO
DISTILLATION_DPO_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
DISTILLATION_DPO_MAX_BACKFILL_ROUNDS ?= 2
DISTILLATION_DPO_PUSH_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_HF_NAMESPACE ?= $(HF_NAMESPACE)
DISTILLATION_DPO_HF_PREFIX ?= slm-synthetic-distillation-dpo
DISTILLATION_DPO_HF_REPO ?= $(DISTILLATION_DPO_HF_NAMESPACE)/slm-synthetic-distillation-dpo
DISTILLATION_DPO_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter command line,$(origin DISTILLATION_DPO_FAMILIES)),$(DISTILLATION_DPO_FAMILIES),$(DISTILLATION_DPO_SMOKE_FAMILIES))

# SFT
SFT_RUN ?= sft-smoke-001
SFT_GENERATION_RUN ?= sft-candidate-001
SFT_REPORT_RUN ?= $(SFT_RUN)
SFT_INSPECT_RUN ?= $(SFT_REPORT_RUN)
SFT_FAMILIES ?= all
SFT_SMOKE_FAMILIES ?= grounded_qa_and_reading
SFT_CANDIDATE_COUNTS ?=
SFT_ACCEPTED_TARGETS ?=
SFT_CANDIDATE_WAVE_SIZE ?= 1000
SFT_SMOKE_CANDIDATE_COUNTS ?= grounded_qa_and_reading=2
SFT_BATCH_SIZE ?= 16
SFT_SMOKE_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)
SFT_CONCURRENCY ?= $(PRETRAIN_CONCURRENCY)
SFT_GENERATION_CONCURRENCY ?= 8
SFT_RUN_ROOT ?= data/sft/runs
SFT_MODEL ?=
SFT_ADJUDICATOR_MODEL ?=
SFT_REVIEWER_MODEL ?=
SFT_INITIAL_CONCURRENCY ?= 8
SFT_INITIAL_BATCH_SIZE ?= 4
SFT_BATCH_INCREASE_SUCCESSES ?= 4
SFT_MAX_TOKENS ?= 4096
SFT_ADJUDICATOR_MAX_TOKENS ?= $(SFT_MAX_TOKENS)
SFT_REVIEWER_MAX_TOKENS ?= 512
SFT_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
SFT_PUSH_RUN ?= $(SFT_REPORT_RUN)
SFT_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(if $(HF_NAMESPACE),$(HF_NAMESPACE)/slm-synthetic-sft,))
SFT_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter file,$(origin SFT_FAMILIES)),$(SFT_SMOKE_FAMILIES),$(SFT_FAMILIES))
SFT_SMOKE_CANDIDATE_COUNTS_EFFECTIVE := $(if $(filter file,$(origin SFT_CANDIDATE_COUNTS)),$(SFT_SMOKE_CANDIDATE_COUNTS),$(SFT_CANDIDATE_COUNTS))

# DPO
DPO_RUN ?= dpo-smoke-001
DPO_GENERATION_RUN ?= dpo-candidate-001
DPO_REPORT_RUN ?= $(DPO_RUN)
DPO_INSPECT_RUN ?= $(DPO_REPORT_RUN)
DPO_PREFERENCE_DIMENSIONS ?= all
DPO_SMOKE_PREFERENCE_DIMENSIONS ?= instruction_adherence
DPO_CANDIDATE_COUNTS ?=
DPO_SMOKE_CANDIDATE_COUNTS ?= instruction_adherence=2
DPO_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)
DPO_SMOKE_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)
DPO_CONCURRENCY ?= $(PRETRAIN_CONCURRENCY)
DPO_GENERATION_CONCURRENCY ?= $(PRETRAIN_TARGET_CONCURRENCY)
DPO_RUN_ROOT ?= data/dpo/runs
DPO_MODEL ?= $(MODEL)
DPO_ADJUDICATOR_MODEL ?= $(DPO_MODEL)
DPO_REVIEWER_MODEL ?= $(DPO_ADJUDICATOR_MODEL)
DPO_INITIAL_CONCURRENCY ?= 8
DPO_INITIAL_BATCH_SIZE ?= 4
DPO_BATCH_INCREASE_SUCCESSES ?= 4
DPO_MAX_TOKENS ?= 4096
DPO_ADJUDICATOR_MAX_TOKENS ?= $(DPO_MAX_TOKENS)
DPO_REVIEWER_MAX_TOKENS ?= 512
DPO_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml
DPO_PUSH_RUN ?= $(DPO_REPORT_RUN)
DPO_HF_REPO ?= $(if $(HF_REPO),$(HF_REPO),$(if $(HF_NAMESPACE),$(HF_NAMESPACE)/slm-synthetic-dpo,))
DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE := $(if $(filter file,$(origin DPO_PREFERENCE_DIMENSIONS)),$(DPO_SMOKE_PREFERENCE_DIMENSIONS),$(DPO_PREFERENCE_DIMENSIONS))
DPO_SMOKE_CANDIDATE_COUNTS_EFFECTIVE := $(if $(filter file,$(origin DPO_CANDIDATE_COUNTS)),$(DPO_SMOKE_CANDIDATE_COUNTS),$(DPO_CANDIDATE_COUNTS))

# Hugging Face dataset deletion
HF_DELETE_NAMESPACE ?= $(HF_NAMESPACE)
HF_DELETE_REPO ?=
HF_DELETE_REPO_FILE ?=
HF_DELETE_YES ?=
HF_DELETE_YES_ARG := $(if $(filter true yes 1,$(HF_DELETE_YES)),--yes,)
HF_DELETE_REPO_ARG := $(if $(HF_DELETE_REPO),--repo $(HF_DELETE_REPO),)
HF_DELETE_REPO_FILE_ARG := $(if $(HF_DELETE_REPO_FILE),--repo-file $(HF_DELETE_REPO_FILE),)

.PHONY: help \
	pretrain-smoke pretrain-generate pretrain-report pretrain-inspect pretrain-push \
	distillation-sft-smoke distillation-sft-generate \
	distillation-sft-report distillation-sft-inspect distillation-sft-adjudicate distillation-sft-push \
	distillation-dpo-smoke distillation-dpo-generate \
	distillation-dpo-report distillation-dpo-inspect distillation-dpo-push \
	alignment-preflight sft-preflight dpo-preflight \
	sft-smoke sft-generate sft-report sft-inspect sft-push \
	dpo-smoke dpo-generate dpo-report dpo-inspect dpo-push \
	model-qualify model-qualify-pretrain model-qualify-alignment \
	model-qualify-distillation model-qualify-all estimate-generation-cost \
	hf-delete-datasets hf-delete-distillation hf-delete-legacy-distillation-dpo \
	test clean

help:
> @echo ""
> @echo "SLM Synthetic Data"
> @echo "=================="
> @echo ""
> @echo "Generate datasets:"
> @echo "  make pretrain-smoke      Small pretraining generation run"
> @echo "  make pretrain-generate   Target pretraining generation run"
> @echo "  make alignment-preflight Validate complete generic SFT and DPO source inventories"
> @echo "  make distillation-sft-smoke       Small distillation SFT run"
> @echo "  make distillation-sft-generate    Target distillation SFT run"
> @echo "  make distillation-dpo-smoke       Small distillation DPO run"
> @echo "  make distillation-dpo-generate    Target distillation DPO run"
> @echo "  make sft-smoke           Small SFT run"
> @echo "  make sft-generate        Target SFT run"
> @echo "  make dpo-smoke           Small DPO run"
> @echo "  make dpo-generate        Target DPO run"
> @echo "  make model-qualify       Qualify one model for selected generation roles"
> @echo "  make estimate-generation-cost  Estimate generator/judge/reviewer cost"
> @echo ""
> @echo "Inspect and report:"
> @echo "  make pretrain-inspect    Show pretraining files and sample rows"
> @echo "  make distillation-sft-inspect     Show distillation SFT files and sample rows"
> @echo "  make distillation-dpo-inspect     Show distillation DPO files and sample rows"
> @echo "  make sft-inspect         Show SFT files and sample rows"
> @echo "  make dpo-inspect         Show DPO files and sample rows"
> @echo "  make pretrain-report     Rebuild pretraining reports and dataset card"
> @echo "  make distillation-sft-report      Rebuild distillation SFT reports and dataset card"
> @echo "  make distillation-sft-adjudicate  Apply reviewed repeated-response decisions"
> @echo "  make distillation-dpo-report      Rebuild distillation DPO reports and dataset card"
> @echo "  make sft-report          Rebuild SFT coverage and dataset card"
> @echo "  make dpo-report          Rebuild DPO coverage and dataset card"
> @echo ""
> @echo "Push to Hugging Face:"
> @echo "  make pretrain-push       Push pretraining deduped data"
> @echo "  make distillation-sft-push        Push a distillation SFT run"
> @echo "  make distillation-dpo-push        Push a distillation DPO run"
> @echo "  make sft-push            Push one consolidated SFT dataset"
> @echo "  make dpo-push            Push one consolidated DPO dataset"
> @echo ""
> @echo "Delete Hugging Face datasets:"
> @echo "  make hf-delete-datasets                Dry-run exact repos from HF_DELETE_REPO/HF_DELETE_REPO_FILE"
> @echo "  make hf-delete-distillation            Dry-run distillation dataset repo deletion"
> @echo "  make hf-delete-legacy-distillation-dpo Dry-run old long distillation-DPO repo deletion"
> @echo "  Set HF_DELETE_YES=1 to actually delete"
> @echo ""
> @echo "Maintenance:"
> @echo "  make test                Run tests"
> @echo "  make clean               Remove generated data"
> @echo ""
> @echo "Common variables:"
> @echo "  MODEL=$(MODEL)"
> @echo "  OPENROUTER_ROUTING_MODE=$(OPENROUTER_ROUTING_MODE)"
> @echo "  OPENROUTER_PROVIDER=$(OPENROUTER_PROVIDER)"
> @echo "  OPENROUTER_PROVIDER_ORDER=$(OPENROUTER_PROVIDER_ORDER)"
> @echo "  OPENROUTER_PROVIDER_ONLY=$(OPENROUTER_PROVIDER_ONLY)"
> @echo "  OPENROUTER_PROVIDER_IGNORE=$(OPENROUTER_PROVIDER_IGNORE)"
> @echo "  OPENROUTER_PROVIDER_SORT=$(OPENROUTER_PROVIDER_SORT)"
> @echo "  PRETRAIN_TOKENS=$(PRETRAIN_TOKENS)"
> @echo "  PRETRAIN_TARGET_TOKENS=$(PRETRAIN_TARGET_TOKENS)"
> @echo "  PRETRAIN_MAX_TOKENS=$(PRETRAIN_MAX_TOKENS)"
> @echo "  PRETRAIN_DIVERSITY_SAMPLE_SIZE=$(PRETRAIN_DIVERSITY_SAMPLE_SIZE)"
> @echo "  PRETRAIN_DIVERSITY_THRESHOLD=$(PRETRAIN_DIVERSITY_THRESHOLD)"
> @echo "  DISTILLATION_SFT_CANDIDATE_COUNTS=$(DISTILLATION_SFT_CANDIDATE_COUNTS)"
> @echo "  DISTILLATION_SFT_CONCURRENCY=$(DISTILLATION_SFT_CONCURRENCY)"
> @echo "  DISTILLATION_SFT_GENERATION_CONCURRENCY=$(DISTILLATION_SFT_GENERATION_CONCURRENCY)"
> @echo "  DISTILLATION_SFT_HF_REPO=$(DISTILLATION_SFT_HF_REPO)"
> @echo "  DISTILLATION_DPO_TARGET_PAIRS=$(DISTILLATION_DPO_TARGET_PAIRS)"
> @echo "  DISTILLATION_DPO_HF_NAMESPACE=$(DISTILLATION_DPO_HF_NAMESPACE)"
> @echo "  DISTILLATION_DPO_HF_REPO=$(DISTILLATION_DPO_HF_REPO)"
> @echo "  SFT_CANDIDATE_COUNTS=$(SFT_CANDIDATE_COUNTS)"
> @echo "  SFT_ACCEPTED_TARGETS=$(SFT_ACCEPTED_TARGETS)"
> @echo "  SFT_CANDIDATE_WAVE_SIZE=$(SFT_CANDIDATE_WAVE_SIZE)"
> @echo "  SFT_HF_REPO=$(SFT_HF_REPO)"
> @echo "  DPO_CANDIDATE_COUNTS=$(DPO_CANDIDATE_COUNTS)"
> @echo "  DPO_HF_REPO=$(DPO_HF_REPO)"
> @echo "  HF_DELETE_NAMESPACE=$(HF_DELETE_NAMESPACE)"
> @echo "  HF_DELETE_REPO=$(HF_DELETE_REPO)"
> @echo "  HF_DELETE_REPO_FILE=$(HF_DELETE_REPO_FILE)"
> @echo "  HF_DELETE_YES=$(HF_DELETE_YES)"
> @echo ""

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
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.pretrain.curate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
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
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.pretrain.curate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.report_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(MAKE) pretrain-report PRETRAIN_REPORT_RUN=$(PRETRAIN_TARGET_RUN)

pretrain-report:
> $(PYTHON) -m slm_synth.pretrain.curate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG) --verify-only
> $(PYTHON) -m slm_synth.pretrain.manifest \
>   --config $(CONFIG_FILE) \
>   --generation-run $(PRETRAIN_REPORT_RUN)
> $(PYTHON) -m slm_synth.pretrain.report_duplicates --config $(CONFIG_FILE) --stage $(PRETRAIN_STAGE)
> $(PYTHON) -m slm_synth.pretrain.report_lengths --config $(CONFIG_FILE) --stage $(PRETRAIN_STAGE)
> $(PYTHON) -m slm_synth.pretrain.report_diversity \
>   --config $(CONFIG_FILE) \
>   --stage $(PRETRAIN_STAGE) \
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
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_sft.cli generate-seed-run \
>   $(DISTILLATION_SFT_SIGNALS_ARG) \
>   --count-per-signal $(DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_RUN)/manifests \
>   --teacher-model $(DISTILLATION_SFT_MODEL) \
>   --generation-run $(DISTILLATION_SFT_RUN) \
>   --max-tokens $(DISTILLATION_SFT_MAX_TOKENS) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --batch-size $(DISTILLATION_SFT_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES)
> $(MAKE) distillation-sft-report DISTILLATION_SFT_REPORT_RUN=$(DISTILLATION_SFT_RUN)

distillation-sft-generate:
> @test -n "$(strip $(DISTILLATION_SFT_CANDIDATE_COUNTS))" || (echo "DISTILLATION_SFT_CANDIDATE_COUNTS is required (signal=count ...)" >&2; exit 2)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_sft.cli generate-production-run \
>   $(DISTILLATION_SFT_SIGNALS_ARG) \
>   --candidate-counts $(DISTILLATION_SFT_CANDIDATE_COUNTS) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_GENERATION_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_GENERATION_RUN)/manifests \
>   --teacher-model $(DISTILLATION_SFT_MODEL) \
>   --generation-run $(DISTILLATION_SFT_GENERATION_RUN) \
>   --max-tokens $(DISTILLATION_SFT_MAX_TOKENS) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --batch-size $(DISTILLATION_SFT_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_GENERATION_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES)
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

distillation-sft-adjudicate:
> test -n "$(DISTILLATION_SFT_ADJUDICATIONS)" || (echo "DISTILLATION_SFT_ADJUDICATIONS is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.distillation_sft.cli apply-response-cluster-adjudications \
>   --dataset-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_ADJUDICATION_RUN)/datasets \
>   --adjudications $(DISTILLATION_SFT_ADJUDICATIONS) \
>   --rejected-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_ADJUDICATION_RUN)/rejected \
>   --run-manifest $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_ADJUDICATION_RUN)/manifests/$(DISTILLATION_SFT_ADJUDICATION_RUN).manifest.json

distillation-sft-push:
> test -n "$(DISTILLATION_SFT_HF_REPO)" || (echo "DISTILLATION_SFT_HF_REPO or HF_REPO is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.distillation_sft.push_hf \
>   --dataset-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN)/datasets \
>   --run-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN) \
>   --repo-id $(DISTILLATION_SFT_HF_REPO) $(HF_PRIVATE_ARG)

distillation-dpo-smoke:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_dpo.cli generate-llm-run \
>   --families $(DISTILLATION_DPO_SMOKE_FAMILIES_EFFECTIVE) \
>   --count-per-family $(DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY) \
>   --batch-size $(DISTILLATION_DPO_BATCH_SIZE) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_RUN)/manifests \
>   --teacher-model $(DISTILLATION_DPO_MODEL) \
>   --generation-run $(DISTILLATION_DPO_RUN) \
>   --max-tokens $(DISTILLATION_DPO_MAX_TOKENS) \
>   --openrouter-routing-mode $(OPENROUTER_ROUTING_MODE) \
>   --concurrency $(DISTILLATION_DPO_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_DPO_BATCH_INCREASE_SUCCESSES) \
>   --max-backfill-rounds $(DISTILLATION_DPO_MAX_BACKFILL_ROUNDS) \
>   --holdout-registry $(DISTILLATION_DPO_HOLDOUT_REGISTRY)
> $(MAKE) distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=$(DISTILLATION_DPO_RUN)

distillation-dpo-generate:
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.distillation_dpo.cli generate-llm-run \
>   --families $(DISTILLATION_DPO_FAMILIES) \
>   --target-pairs $(DISTILLATION_DPO_TARGET_PAIRS) \
>   --batch-size $(DISTILLATION_DPO_BATCH_SIZE) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_TARGET_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_TARGET_RUN)/manifests \
>   --teacher-model $(DISTILLATION_DPO_MODEL) \
>   --generation-run $(DISTILLATION_DPO_TARGET_RUN) \
>   --max-tokens $(DISTILLATION_DPO_MAX_TOKENS) \
>   --openrouter-routing-mode $(OPENROUTER_ROUTING_MODE) \
>   --concurrency $(DISTILLATION_DPO_TARGET_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_DPO_BATCH_INCREASE_SUCCESSES) \
>   --max-backfill-rounds $(DISTILLATION_DPO_MAX_BACKFILL_ROUNDS) \
>   --holdout-registry $(DISTILLATION_DPO_HOLDOUT_REGISTRY)
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

alignment-preflight:
> $(PYTHON) -m slm_synth.alignment_preflight --kind all

sft-preflight:
> $(PYTHON) -m slm_synth.alignment_preflight --kind sft

dpo-preflight:
> $(PYTHON) -m slm_synth.alignment_preflight --kind dpo

sft-smoke:
> @test -n "$(strip $(SFT_MODEL))" || (echo "SFT_MODEL is required" >&2; exit 2)
> @test -n "$(strip $(SFT_ADJUDICATOR_MODEL))" || (echo "SFT_ADJUDICATOR_MODEL is required" >&2; exit 2)
> @test -n "$(strip $(SFT_REVIEWER_MODEL))" || (echo "SFT_REVIEWER_MODEL is required" >&2; exit 2)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_SMOKE_FAMILIES_EFFECTIVE) \
>   --candidate-counts $(SFT_SMOKE_CANDIDATE_COUNTS_EFFECTIVE) \
>   --batch-size $(SFT_SMOKE_BATCH_SIZE) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_RUN)/datasets \
>   --manifest-dir $(SFT_RUN_ROOT)/$(SFT_RUN)/manifests \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --adjudicator-model $(SFT_ADJUDICATOR_MODEL) \
>   --adjudicator-max-tokens $(SFT_ADJUDICATOR_MAX_TOKENS) \
>   --reviewer-model $(SFT_REVIEWER_MODEL) \
>   --reviewer-max-tokens $(SFT_REVIEWER_MAX_TOKENS) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --concurrency $(SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(SFT_BATCH_INCREASE_SUCCESSES) \
>   --holdout-registry $(SFT_HOLDOUT_REGISTRY)
> $(MAKE) sft-report SFT_REPORT_RUN=$(SFT_RUN)

sft-generate:
> @test -n "$(strip $(SFT_MODEL))" || (echo "SFT_MODEL is required" >&2; exit 2)
> @test -n "$(strip $(SFT_ADJUDICATOR_MODEL))" || (echo "SFT_ADJUDICATOR_MODEL is required" >&2; exit 2)
> @test -n "$(strip $(SFT_REVIEWER_MODEL))" || (echo "SFT_REVIEWER_MODEL is required" >&2; exit 2)
> @test -n "$(strip $(SFT_CANDIDATE_COUNTS))$(strip $(SFT_ACCEPTED_TARGETS))" || (echo "set exactly one of SFT_CANDIDATE_COUNTS or SFT_ACCEPTED_TARGETS" >&2; exit 2)
> @test -z "$(strip $(SFT_CANDIDATE_COUNTS))" -o -z "$(strip $(SFT_ACCEPTED_TARGETS))" || (echo "SFT_CANDIDATE_COUNTS and SFT_ACCEPTED_TARGETS are mutually exclusive" >&2; exit 2)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_FAMILIES) \
>   $(if $(strip $(SFT_ACCEPTED_TARGETS)),--accepted-targets $(SFT_ACCEPTED_TARGETS) --candidate-wave-size $(SFT_CANDIDATE_WAVE_SIZE),--candidate-counts $(SFT_CANDIDATE_COUNTS)) \
>   --batch-size $(SFT_BATCH_SIZE) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_GENERATION_RUN)/datasets \
>   --manifest-dir $(SFT_RUN_ROOT)/$(SFT_GENERATION_RUN)/manifests \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --adjudicator-model $(SFT_ADJUDICATOR_MODEL) \
>   --adjudicator-max-tokens $(SFT_ADJUDICATOR_MAX_TOKENS) \
>   --reviewer-model $(SFT_REVIEWER_MODEL) \
>   --reviewer-max-tokens $(SFT_REVIEWER_MAX_TOKENS) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --concurrency $(SFT_GENERATION_CONCURRENCY) \
>   --adaptive-initial-in-flight $(SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(SFT_BATCH_INCREASE_SUCCESSES) \
>   --holdout-registry $(SFT_HOLDOUT_REGISTRY)
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
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --preference-dimensions $(DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE) \
>   --candidate-counts $(DPO_SMOKE_CANDIDATE_COUNTS_EFFECTIVE) \
>   --batch-size $(DPO_SMOKE_BATCH_SIZE) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_RUN)/datasets \
>   --manifest-dir $(DPO_RUN_ROOT)/$(DPO_RUN)/manifests \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --adjudicator-model $(DPO_ADJUDICATOR_MODEL) \
>   --adjudicator-max-tokens $(DPO_ADJUDICATOR_MAX_TOKENS) \
>   --reviewer-model $(DPO_REVIEWER_MODEL) \
>   --reviewer-max-tokens $(DPO_REVIEWER_MAX_TOKENS) \
>   --holdout-registry $(DPO_HOLDOUT_REGISTRY) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --concurrency $(DPO_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DPO_BATCH_INCREASE_SUCCESSES)
> $(MAKE) dpo-report DPO_REPORT_RUN=$(DPO_RUN)

dpo-generate:
> @test -n "$(strip $(DPO_CANDIDATE_COUNTS))" || (echo "DPO_CANDIDATE_COUNTS is required (dimension=count ...)" >&2; exit 2)
> $(OPENROUTER_ENV) $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --preference-dimensions $(DPO_PREFERENCE_DIMENSIONS) \
>   --candidate-counts $(DPO_CANDIDATE_COUNTS) \
>   --batch-size $(DPO_BATCH_SIZE) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_GENERATION_RUN)/datasets \
>   --manifest-dir $(DPO_RUN_ROOT)/$(DPO_GENERATION_RUN)/manifests \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --adjudicator-model $(DPO_ADJUDICATOR_MODEL) \
>   --adjudicator-max-tokens $(DPO_ADJUDICATOR_MAX_TOKENS) \
>   --reviewer-model $(DPO_REVIEWER_MODEL) \
>   --reviewer-max-tokens $(DPO_REVIEWER_MAX_TOKENS) \
>   --holdout-registry $(DPO_HOLDOUT_REGISTRY) \
>   $(OPENROUTER_ROUTING_ARGS) \
>   --concurrency $(DPO_GENERATION_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DPO_BATCH_INCREASE_SUCCESSES)
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
>   --output $(QUALIFY_OUTPUT)

model-qualify-pretrain:
> $(MAKE) model-qualify QUALIFY_ROLES=pretrain-generator

model-qualify-alignment:
> $(MAKE) model-qualify QUALIFY_ROLES=sft-generator,sft-judge,sft-reviewer,dpo-generator,dpo-judge,dpo-reviewer

model-qualify-distillation:
> $(MAKE) model-qualify QUALIFY_ROLES=distillation-sft-generator,distillation-sft-judge,distillation-sft-reviewer,distillation-dpo-generator,distillation-dpo-judge,distillation-dpo-reviewer

model-qualify-all:
> $(MAKE) model-qualify QUALIFY_ROLES=all

estimate-generation-cost:
> $(PYTHON) -m slm_synth.estimate_generation_cost \
>   --generator-model $(COST_GENERATOR_MODEL) \
>   --judge-model $(COST_JUDGE_MODEL) \
>   --reviewer-model $(COST_REVIEWER_MODEL) \
>   --candidates $(COST_CANDIDATES) \
>   --average-accepted-tokens $(COST_AVERAGE_ACCEPTED_TOKENS) \
>   $(COST_TARGET_ACCEPTED_ARG) \
>   $(COST_TARGET_TOKENS_ARG) \
>   $(COST_OUTPUT_ARG)

hf-delete-datasets:
> $(PYTHON) scripts/delete_hf_datasets.py \
>   --namespace $(HF_DELETE_NAMESPACE) \
>   $(HF_DELETE_REPO_ARG) \
>   $(HF_DELETE_REPO_FILE_ARG) \
>   $(HF_DELETE_YES_ARG)

hf-delete-distillation:
> $(PYTHON) scripts/delete_hf_datasets.py \
>   --namespace $(HF_DELETE_NAMESPACE) \
>   --include-distillation \
>   $(HF_DELETE_YES_ARG)

hf-delete-legacy-distillation-dpo:
> $(PYTHON) scripts/delete_hf_datasets.py \
>   --namespace $(HF_DELETE_NAMESPACE) \
>   --include-legacy-distillation-dpo \
>   $(HF_DELETE_YES_ARG)

test:
> $(PYTHON) -m compileall -q slm_synth tests
> pytest -q

clean:
> rm -rf $(DATA_DIR) data/distillation data/sft data/dpo
