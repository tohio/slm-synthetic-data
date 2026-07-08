# ============================================================
# SLM Synthetic Data
# ============================================================

.RECIPEPREFIX := >
MAKEFLAGS += --no-print-directory

PYTHON ?= python

# Shared defaults
MODEL ?= openai/gpt-4.1-mini
MAX_TOKENS ?= 4096

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
PRETRAIN_CONCURRENCY ?= 1
PRETRAIN_TARGET_CONCURRENCY ?= 4
PRETRAIN_MODEL ?= $(MODEL)
PRETRAIN_SIGNAL ?=
PRETRAIN_SIGNAL_ARG := $(if $(PRETRAIN_SIGNAL),--signal $(PRETRAIN_SIGNAL),)
PRETRAIN_STAGE ?= deduped
HF_REPO ?=
HF_NAMESPACE ?= tohio
HF_PRIVATE ?=
HF_PRIVATE_ARG := $(if $(filter true yes 1,$(HF_PRIVATE)),--private,)

# Distillation SFT
DISTILLATION_SFT_RUN ?= distillation-sft-smoke-001
DISTILLATION_SFT_TARGET_RUN ?= distillation-sft-target-001
DISTILLATION_SFT_REPORT_RUN ?= $(DISTILLATION_SFT_RUN)
DISTILLATION_SFT_INSPECT_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_TARGET_ROWS ?= 100000
DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL ?= 2
DISTILLATION_SFT_BATCH_SIZE ?= 5
DISTILLATION_SFT_CONCURRENCY ?= 1
DISTILLATION_SFT_SIGNALS ?=
DISTILLATION_SFT_SIGNALS_ARG := $(if $(filter all,$(DISTILLATION_SFT_SIGNALS)),,$(if $(DISTILLATION_SFT_SIGNALS),--signals $(DISTILLATION_SFT_SIGNALS),))
DISTILLATION_SFT_INITIAL_CONCURRENCY ?= 8
DISTILLATION_SFT_INITIAL_BATCH_SIZE ?= 4
DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES ?= 16
DISTILLATION_SFT_RUN_ROOT ?= data/distillation/runs
DISTILLATION_SFT_MODEL ?= $(MODEL)
DISTILLATION_SFT_MAX_TOKENS ?= 4096
DISTILLATION_SFT_DATASET_NAME ?= SLM Synthetic Distillation
DISTILLATION_SFT_PUSH_RUN ?= $(DISTILLATION_SFT_REPORT_RUN)
DISTILLATION_SFT_HF_REPO ?= $(HF_REPO)

# Distillation DPO
DISTILLATION_DPO_RUN ?= distillation-dpo-smoke-001
DISTILLATION_DPO_TARGET_RUN ?= distillation-dpo-target-001
DISTILLATION_DPO_REPORT_RUN ?= $(DISTILLATION_DPO_RUN)
DISTILLATION_DPO_INSPECT_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_FAMILIES ?= all
DISTILLATION_DPO_SMOKE_FAMILIES ?= teacher_response_preference
DISTILLATION_DPO_COUNT_PER_FAMILY ?= 1000
DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY ?= 2
DISTILLATION_DPO_RUN_ROOT ?= data/distillation-dpo/runs
DISTILLATION_DPO_MODEL ?= $(MODEL)
DISTILLATION_DPO_DATASET_NAME ?= SLM Synthetic Distillation DPO
DISTILLATION_DPO_PUSH_RUN ?= $(DISTILLATION_DPO_REPORT_RUN)
DISTILLATION_DPO_HF_NAMESPACE ?= $(HF_NAMESPACE)
DISTILLATION_DPO_HF_PREFIX ?= distillation-dpo
DISTILLATION_DPO_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter command line,$(origin DISTILLATION_DPO_FAMILIES)),$(DISTILLATION_DPO_FAMILIES),$(DISTILLATION_DPO_SMOKE_FAMILIES))

# SFT
SFT_RUN ?= sft-smoke-001
SFT_TARGET_RUN ?= sft-target-001
SFT_REPORT_RUN ?= $(SFT_RUN)
SFT_INSPECT_RUN ?= $(SFT_REPORT_RUN)
SFT_FAMILIES ?= all
SFT_SMOKE_FAMILIES ?= basic_arithmetic_qa
SFT_COUNT_PER_FAMILY ?= 1000
SFT_SMOKE_COUNT_PER_FAMILY ?= 2
SFT_BATCH_SIZE ?= 5
SFT_SMOKE_BATCH_SIZE ?= 2
SFT_CONCURRENCY ?= 1
SFT_RUN_ROOT ?= data/sft/runs
SFT_MODEL ?= $(MODEL)
SFT_INITIAL_CONCURRENCY ?= 8
SFT_INITIAL_BATCH_SIZE ?= 4
SFT_BATCH_INCREASE_SUCCESSES ?= 16
SFT_MAX_TOKENS ?= 4096
SFT_PUSH_RUN ?= $(SFT_REPORT_RUN)
SFT_HF_REPO ?= $(HF_REPO)
SFT_HF_NAMESPACE ?= $(HF_NAMESPACE)
SFT_HF_PREFIX ?= slm-sft
SFT_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter command line,$(origin SFT_FAMILIES)),$(SFT_FAMILIES),$(SFT_SMOKE_FAMILIES))

# DPO
DPO_RUN ?= dpo-smoke-001
DPO_TARGET_RUN ?= dpo-target-001
DPO_REPORT_RUN ?= $(DPO_RUN)
DPO_INSPECT_RUN ?= $(DPO_REPORT_RUN)
DPO_FAMILIES ?= all
DPO_SMOKE_FAMILIES ?= basic_arithmetic_qa
DPO_COUNT_PER_FAMILY ?= 1000
DPO_SMOKE_COUNT_PER_FAMILY ?= 2
DPO_BATCH_SIZE ?= 5
DPO_SMOKE_BATCH_SIZE ?= 2
DPO_CONCURRENCY ?= 1
DPO_RUN_ROOT ?= data/dpo/runs
DPO_MODEL ?= $(MODEL)
DPO_INITIAL_CONCURRENCY ?= 8
DPO_INITIAL_BATCH_SIZE ?= 4
DPO_BATCH_INCREASE_SUCCESSES ?= 16
DPO_MAX_TOKENS ?= 4096
DPO_PUSH_RUN ?= $(DPO_REPORT_RUN)
DPO_HF_REPO ?= $(HF_REPO)
DPO_HF_NAMESPACE ?= $(HF_NAMESPACE)
DPO_HF_PREFIX ?= slm-dpo
DPO_SMOKE_FAMILIES_EFFECTIVE := $(if $(filter command line,$(origin DPO_FAMILIES)),$(DPO_FAMILIES),$(DPO_SMOKE_FAMILIES))

.PHONY: help \
	pretrain-smoke pretrain-generate pretrain-report pretrain-inspect pretrain-push \
	distillation-sft-smoke distillation-sft-generate \
	distillation-sft-report distillation-sft-inspect distillation-sft-push \
	distillation-dpo-smoke distillation-dpo-generate \
	distillation-dpo-report distillation-dpo-inspect distillation-dpo-push \
	sft-smoke sft-generate sft-report sft-inspect sft-push \
	dpo-smoke dpo-generate dpo-report dpo-inspect dpo-push \
	test clean

help:
> @echo ""
> @echo "SLM Synthetic Data"
> @echo "=================="
> @echo ""
> @echo "Generate datasets:"
> @echo "  make pretrain-smoke      Small pretraining generation run"
> @echo "  make pretrain-generate   Target pretraining generation run"
> @echo "  make distillation-sft-smoke       Small distillation SFT run"
> @echo "  make distillation-sft-generate    Target distillation SFT run"
> @echo "  make distillation-dpo-smoke       Small distillation DPO run"
> @echo "  make distillation-dpo-generate    Target distillation DPO run"
> @echo "  make sft-smoke           Small SFT run"
> @echo "  make sft-generate        Target SFT run"
> @echo "  make dpo-smoke           Small DPO run"
> @echo "  make dpo-generate        Target DPO run"
> @echo ""
> @echo "Inspect and report:"
> @echo "  make pretrain-inspect    Show pretraining files and sample rows"
> @echo "  make distillation-sft-inspect     Show distillation SFT files and sample rows"
> @echo "  make distillation-dpo-inspect     Show distillation DPO files and sample rows"
> @echo "  make sft-inspect         Show SFT files and sample rows"
> @echo "  make dpo-inspect         Show DPO files and sample rows"
> @echo "  make pretrain-report     Rebuild pretraining reports"
> @echo "  make distillation-sft-report      Rebuild distillation SFT reports and dataset card"
> @echo "  make distillation-dpo-report      Rebuild distillation DPO reports and dataset card"
> @echo "  make sft-report          Rebuild SFT coverage"
> @echo "  make dpo-report          Rebuild DPO coverage"
> @echo ""
> @echo "Push to Hugging Face:"
> @echo "  make pretrain-push       Push pretraining deduped data"
> @echo "  make distillation-sft-push        Push a distillation SFT run"
> @echo "  make distillation-dpo-push        Push distillation DPO families"
> @echo "  make sft-push            Push SFT families to slm-sft-* repos"
> @echo "  make dpo-push            Push DPO families to slm-dpo-* repos"
> @echo ""
> @echo "Maintenance:"
> @echo "  make test                Run tests"
> @echo "  make clean               Remove generated data"
> @echo ""
> @echo "Common variables:"
> @echo "  MODEL=$(MODEL)"
> @echo "  PRETRAIN_TOKENS=$(PRETRAIN_TOKENS)"
> @echo "  PRETRAIN_TARGET_TOKENS=$(PRETRAIN_TARGET_TOKENS)"
> @echo "  DISTILLATION_SFT_TARGET_ROWS=$(DISTILLATION_SFT_TARGET_ROWS)"
> @echo "  DISTILLATION_SFT_CONCURRENCY=$(DISTILLATION_SFT_CONCURRENCY)"
> @echo "  DISTILLATION_SFT_HF_REPO=$(DISTILLATION_SFT_HF_REPO)"
> @echo "  DISTILLATION_DPO_COUNT_PER_FAMILY=$(DISTILLATION_DPO_COUNT_PER_FAMILY)"
> @echo "  DISTILLATION_DPO_HF_NAMESPACE=$(DISTILLATION_DPO_HF_NAMESPACE)"
> @echo "  DISTILLATION_DPO_HF_PREFIX=$(DISTILLATION_DPO_HF_PREFIX)"
> @echo "  SFT_COUNT_PER_FAMILY=$(SFT_COUNT_PER_FAMILY)"
> @echo "  SFT_HF_NAMESPACE=$(SFT_HF_NAMESPACE)"
> @echo "  SFT_HF_PREFIX=$(SFT_HF_PREFIX)"
> @echo "  DPO_COUNT_PER_FAMILY=$(DPO_COUNT_PER_FAMILY)"
> @echo "  DPO_HF_NAMESPACE=$(DPO_HF_NAMESPACE)"
> @echo "  DPO_HF_PREFIX=$(DPO_HF_PREFIX)"
> @echo ""

pretrain-smoke:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(PRETRAIN_TOKENS) \
>   --batch-size $(PRETRAIN_BATCH_SIZE) \
>   --model $(PRETRAIN_MODEL) \
>   --concurrency $(PRETRAIN_CONCURRENCY) \
>   --run $(PRETRAIN_RUN) \
>   $(if $(HF_REPO),--hf_repo $(HF_REPO),)
> $(PYTHON) -m slm_synth.pretrain.preflight_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.generate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.report_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.validate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.dedup --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(MAKE) pretrain-report PRETRAIN_REPORT_RUN=$(PRETRAIN_RUN)

pretrain-generate:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(PRETRAIN_TARGET_TOKENS) \
>   --batch-size $(PRETRAIN_BATCH_SIZE) \
>   --model $(PRETRAIN_MODEL) \
>   --concurrency $(PRETRAIN_TARGET_CONCURRENCY) \
>   --run $(PRETRAIN_TARGET_RUN) \
>   $(if $(HF_REPO),--hf_repo $(HF_REPO),)
> $(PYTHON) -m slm_synth.pretrain.preflight_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.generate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.report_artifacts --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.validate --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(PYTHON) -m slm_synth.pretrain.dedup --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)
> $(MAKE) pretrain-report PRETRAIN_REPORT_RUN=$(PRETRAIN_TARGET_RUN)

pretrain-report:
> $(PYTHON) -m slm_synth.pretrain.manifest \
>   --config $(CONFIG_FILE) \
>   --generation-run $(PRETRAIN_REPORT_RUN)
> $(PYTHON) -m slm_synth.pretrain.report_duplicates --config $(CONFIG_FILE) --stage $(PRETRAIN_STAGE)
> $(PYTHON) -m slm_synth.pretrain.report_lengths --config $(CONFIG_FILE) --stage $(PRETRAIN_STAGE)

pretrain-inspect:
> @echo "== pretraining files =="
> @find $(DATA_DIR)/$(PRETRAIN_INSPECT_RUN) -type f 2>/dev/null | sort | tail -n 50
> @echo "== pretraining sample rows =="
> @find $(DATA_DIR)/$(PRETRAIN_INSPECT_RUN) -path '*/deduped/*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

pretrain-push:
> $(PYTHON) -m slm_synth.pretrain.push_hf --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)

distillation-sft-smoke:
> $(PYTHON) -m slm_synth.distillation.cli generate-seed-run \
>   $(DISTILLATION_SFT_SIGNALS_ARG) \
>   --count-per-signal $(DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_RUN)/manifests \
>   --teacher-model $(DISTILLATION_SFT_MODEL) \
>   --generation-run $(DISTILLATION_SFT_RUN) \
>   --max-tokens $(DISTILLATION_SFT_MAX_TOKENS) \
>   --batch-size $(DISTILLATION_SFT_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES)
> $(MAKE) distillation-sft-report DISTILLATION_SFT_REPORT_RUN=$(DISTILLATION_SFT_RUN)

distillation-sft-generate:
> $(PYTHON) -m slm_synth.distillation.cli generate-production-run \
>   $(DISTILLATION_SFT_SIGNALS_ARG) \
>   --target-rows $(DISTILLATION_SFT_TARGET_ROWS) \
>   --output-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_TARGET_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_TARGET_RUN)/manifests \
>   --teacher-model $(DISTILLATION_SFT_MODEL) \
>   --generation-run $(DISTILLATION_SFT_TARGET_RUN) \
>   --max-tokens $(DISTILLATION_SFT_MAX_TOKENS) \
>   --batch-size $(DISTILLATION_SFT_BATCH_SIZE) \
>   --concurrency $(DISTILLATION_SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DISTILLATION_SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DISTILLATION_SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DISTILLATION_SFT_BATCH_INCREASE_SUCCESSES)
> $(MAKE) distillation-sft-report DISTILLATION_SFT_REPORT_RUN=$(DISTILLATION_SFT_TARGET_RUN)

distillation-sft-report:
> $(PYTHON) -m slm_synth.distillation.cli report-coverage \
>   --run-manifest $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/manifests/$(DISTILLATION_SFT_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.distillation.cli build-dataset-card \
>   --run-manifest $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/manifests/$(DISTILLATION_SFT_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_REPORT_RUN)/README.md \
>   --dataset-name "$(DISTILLATION_SFT_DATASET_NAME)"

distillation-sft-inspect:
> @echo "== distillation files =="
> @find $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== distillation sample rows =="
> @find $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

distillation-sft-push:
> test -n "$(DISTILLATION_SFT_HF_REPO)" || (echo "DISTILLATION_SFT_HF_REPO or HF_REPO is required" >&2; exit 2)
> $(PYTHON) -m slm_synth.distillation.push_hf \
>   --dataset-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN)/datasets \
>   --run-dir $(DISTILLATION_SFT_RUN_ROOT)/$(DISTILLATION_SFT_PUSH_RUN) \
>   --repo-id $(DISTILLATION_SFT_HF_REPO) $(HF_PRIVATE_ARG)

distillation-dpo-smoke:
> $(PYTHON) -m slm_synth.distillation_dpo.cli materialize-seed-run \
>   --families $(DISTILLATION_DPO_SMOKE_FAMILIES_EFFECTIVE) \
>   --count-per-family $(DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_RUN)/manifests \
>   --teacher-model $(DISTILLATION_DPO_MODEL) \
>   --generation-run $(DISTILLATION_DPO_RUN)
> $(MAKE) distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=$(DISTILLATION_DPO_RUN)

distillation-dpo-generate:
> $(PYTHON) -m slm_synth.distillation_dpo.cli materialize-seed-run \
>   --families $(DISTILLATION_DPO_FAMILIES) \
>   --count-per-family $(DISTILLATION_DPO_COUNT_PER_FAMILY) \
>   --output-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_TARGET_RUN)/datasets \
>   --manifest-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_TARGET_RUN)/manifests \
>   --teacher-model $(DISTILLATION_DPO_MODEL) \
>   --generation-run $(DISTILLATION_DPO_TARGET_RUN)
> $(MAKE) distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=$(DISTILLATION_DPO_TARGET_RUN)

distillation-dpo-report:
> $(PYTHON) -m slm_synth.distillation_dpo.cli report-coverage \
>   --input $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/datasets \
>   --output $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/coverage.json
> $(PYTHON) -m slm_synth.distillation_dpo.cli build-dataset-card \
>   --run-manifest $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/manifests/$(DISTILLATION_DPO_REPORT_RUN).manifest.json \
>   --output $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_REPORT_RUN)/README.md \
>   --dataset-name "$(DISTILLATION_DPO_DATASET_NAME)"

distillation-dpo-inspect:
> @echo "== distillation DPO files =="
> @find $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== distillation DPO sample rows =="
> @find $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

distillation-dpo-push:
> $(PYTHON) -m slm_synth.distillation_dpo.push_hf \
>   --dataset-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_PUSH_RUN)/datasets \
>   --run-dir $(DISTILLATION_DPO_RUN_ROOT)/$(DISTILLATION_DPO_PUSH_RUN) \
>   --repo-owner $(DISTILLATION_DPO_HF_NAMESPACE) \
>   --repo-prefix $(DISTILLATION_DPO_HF_PREFIX) $(HF_PRIVATE_ARG)

sft-smoke:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_SMOKE_FAMILIES_EFFECTIVE) \
>   --count-per-family $(SFT_SMOKE_COUNT_PER_FAMILY) \
>   --batch-size $(SFT_SMOKE_BATCH_SIZE) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_RUN)/datasets \
>   --manifest-dir $(SFT_RUN_ROOT)/$(SFT_RUN)/manifests \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --concurrency $(SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(SFT_BATCH_INCREASE_SUCCESSES)
> $(MAKE) sft-report SFT_REPORT_RUN=$(SFT_RUN)

sft-generate:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_FAMILIES) \
>   --count-per-family $(SFT_COUNT_PER_FAMILY) \
>   --batch-size $(SFT_BATCH_SIZE) \
>   --output-dir $(SFT_RUN_ROOT)/$(SFT_TARGET_RUN)/datasets \
>   --manifest-dir $(SFT_RUN_ROOT)/$(SFT_TARGET_RUN)/manifests \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_TARGET_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --concurrency $(SFT_CONCURRENCY) \
>   --adaptive-initial-in-flight $(SFT_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(SFT_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(SFT_BATCH_INCREASE_SUCCESSES)
> $(MAKE) sft-report SFT_REPORT_RUN=$(SFT_TARGET_RUN)

sft-report:
> $(PYTHON) -m slm_synth.sft.cli report-coverage \
>   --input $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/datasets \
>   --output $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/coverage.json

sft-inspect:
> @echo "== SFT files =="
> @find $(SFT_RUN_ROOT)/$(SFT_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== SFT sample rows =="
> @find $(SFT_RUN_ROOT)/$(SFT_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

sft-push:
> $(PYTHON) -m slm_synth.sft.push_hf \
>   --dataset-dir $(SFT_RUN_ROOT)/$(SFT_PUSH_RUN)/datasets \
>   --run-dir $(SFT_RUN_ROOT)/$(SFT_PUSH_RUN) \
>   --repo-owner $(SFT_HF_NAMESPACE) \
>   --repo-prefix $(SFT_HF_PREFIX) $(HF_PRIVATE_ARG)

dpo-smoke:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --families $(DPO_SMOKE_FAMILIES_EFFECTIVE) \
>   --count-per-family $(DPO_SMOKE_COUNT_PER_FAMILY) \
>   --batch-size $(DPO_SMOKE_BATCH_SIZE) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_RUN)/datasets \
>   --manifest-dir $(DPO_RUN_ROOT)/$(DPO_RUN)/manifests \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --concurrency $(DPO_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DPO_BATCH_INCREASE_SUCCESSES)
> $(MAKE) dpo-report DPO_REPORT_RUN=$(DPO_RUN)

dpo-generate:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --families $(DPO_FAMILIES) \
>   --count-per-family $(DPO_COUNT_PER_FAMILY) \
>   --batch-size $(DPO_BATCH_SIZE) \
>   --output-dir $(DPO_RUN_ROOT)/$(DPO_TARGET_RUN)/datasets \
>   --manifest-dir $(DPO_RUN_ROOT)/$(DPO_TARGET_RUN)/manifests \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_TARGET_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --concurrency $(DPO_CONCURRENCY) \
>   --adaptive-initial-in-flight $(DPO_INITIAL_CONCURRENCY) \
>   --adaptive-initial-batch-size $(DPO_INITIAL_BATCH_SIZE) \
>   --adaptive-batch-increase-successes $(DPO_BATCH_INCREASE_SUCCESSES)
> $(MAKE) dpo-report DPO_REPORT_RUN=$(DPO_TARGET_RUN)

dpo-report:
> $(PYTHON) -m slm_synth.dpo.cli report-coverage \
>   --input $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/datasets \
>   --output $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/coverage.json

dpo-inspect:
> @echo "== DPO files =="
> @find $(DPO_RUN_ROOT)/$(DPO_INSPECT_RUN) -type f 2>/dev/null | sort
> @echo "== DPO sample rows =="
> @find $(DPO_RUN_ROOT)/$(DPO_INSPECT_RUN)/datasets -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

dpo-push:
> $(PYTHON) -m slm_synth.dpo.push_hf \
>   --dataset-dir $(DPO_RUN_ROOT)/$(DPO_PUSH_RUN)/datasets \
>   --run-dir $(DPO_RUN_ROOT)/$(DPO_PUSH_RUN) \
>   --repo-owner $(DPO_HF_NAMESPACE) \
>   --repo-prefix $(DPO_HF_PREFIX) $(HF_PRIVATE_ARG)

test:
> $(PYTHON) -m compileall -q slm_synth tests
> pytest -q

clean:
> rm -rf $(DATA_DIR) data/distillation data/sft data/dpo
