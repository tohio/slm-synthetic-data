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

# Distillation
DISTILL_RUN ?= distill-smoke-001
DISTILL_TARGET_RUN ?= distill-target-001
DISTILL_REPORT_RUN ?= $(DISTILL_RUN)
DISTILL_TARGET ?= smoke
DISTILL_TARGET_SIZE ?= pilot
DISTILL_SMOKE_COUNT_PER_SIGNAL ?= 2
DISTILL_BATCH_SIZE ?= 5
DISTILL_SIGNALS ?=
DISTILL_SIGNALS_ARG := $(if $(DISTILL_SIGNALS),--signals $(DISTILL_SIGNALS),)
DISTILL_ESTIMATED_TOKENS_PER_ROW ?= 512
DISTILL_OUTPUT_DIR ?= data/distillation/datasets
DISTILL_MANIFEST_DIR ?= data/distillation/manifests
DISTILL_MODEL ?= $(MODEL)
DISTILL_MAX_TOKENS ?= 4096
DISTILL_COVERAGE_REPORT ?= data/distillation/coverage.json
DISTILL_DATASET_CARD ?= data/distillation/README.md
DISTILL_DATASET_NAME ?= SLM Synthetic Distillation

# SFT
SFT_RUN ?= sft-smoke-001
SFT_TARGET_RUN ?= sft-target-001
SFT_FAMILIES ?= all
SFT_SMOKE_FAMILIES ?= basic_arithmetic_qa
SFT_COUNT_PER_FAMILY ?= 1000
SFT_SMOKE_COUNT_PER_FAMILY ?= 2
SFT_BATCH_SIZE ?= 5
SFT_SMOKE_BATCH_SIZE ?= 2
SFT_CONCURRENCY ?= 1
SFT_OUTPUT_DIR ?= data/sft/datasets
SFT_MANIFEST_DIR ?= data/sft/manifests
SFT_COVERAGE_REPORT ?= data/sft/coverage.json
SFT_MODEL ?= $(MODEL)
SFT_MAX_TOKENS ?= 4096

# DPO
DPO_RUN ?= dpo-smoke-001
DPO_TARGET_RUN ?= dpo-target-001
DPO_FAMILIES ?= all
DPO_SMOKE_FAMILIES ?= basic_arithmetic_qa
DPO_COUNT_PER_FAMILY ?= 1000
DPO_SMOKE_COUNT_PER_FAMILY ?= 2
DPO_BATCH_SIZE ?= 5
DPO_SMOKE_BATCH_SIZE ?= 2
DPO_CONCURRENCY ?= 1
DPO_OUTPUT_DIR ?= data/dpo/datasets
DPO_MANIFEST_DIR ?= data/dpo/manifests
DPO_COVERAGE_REPORT ?= data/dpo/coverage.json
DPO_MODEL ?= $(MODEL)
DPO_MAX_TOKENS ?= 4096

.PHONY: help \
	pretrain-smoke pretrain-generate pretrain-report pretrain-inspect pretrain-push \
	distill-smoke distill-generate distill-report distill-inspect \
	sft-smoke sft-generate sft-report sft-inspect \
	dpo-smoke dpo-generate dpo-report dpo-inspect \
	test clean

help:
> @echo ""
> @echo "SLM Synthetic Data"
> @echo "=================="
> @echo ""
> @echo "Generate datasets:"
> @echo "  make pretrain-smoke      Small pretraining generation run"
> @echo "  make pretrain-generate   Target pretraining generation run"
> @echo "  make distill-smoke       Small response distillation run"
> @echo "  make distill-generate    Target response distillation run"
> @echo "  make sft-smoke           Small SFT run"
> @echo "  make sft-generate        Target SFT run"
> @echo "  make dpo-smoke           Small DPO run"
> @echo "  make dpo-generate        Target DPO run"
> @echo ""
> @echo "Inspect and report:"
> @echo "  make pretrain-inspect    Show pretraining files and sample rows"
> @echo "  make distill-inspect     Show distillation files and sample rows"
> @echo "  make sft-inspect         Show SFT files and sample rows"
> @echo "  make dpo-inspect         Show DPO files and sample rows"
> @echo "  make pretrain-report     Rebuild pretraining reports"
> @echo "  make distill-report      Rebuild distillation reports and dataset card"
> @echo "  make sft-report          Rebuild SFT coverage"
> @echo "  make dpo-report          Rebuild DPO coverage"
> @echo ""
> @echo "Maintenance:"
> @echo "  make test                Run tests"
> @echo "  make clean               Remove generated data"
> @echo ""
> @echo "Common variables:"
> @echo "  MODEL=$(MODEL)"
> @echo "  PRETRAIN_TOKENS=$(PRETRAIN_TOKENS)"
> @echo "  PRETRAIN_TARGET_TOKENS=$(PRETRAIN_TARGET_TOKENS)"
> @echo "  DISTILL_TARGET=$(DISTILL_TARGET)"
> @echo "  DISTILL_TARGET_SIZE=$(DISTILL_TARGET_SIZE)"
> @echo "  SFT_COUNT_PER_FAMILY=$(SFT_COUNT_PER_FAMILY)"
> @echo "  DPO_COUNT_PER_FAMILY=$(DPO_COUNT_PER_FAMILY)"
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
> @find $(DATA_DIR) -type f | sort | tail -n 50
> @echo "== pretraining sample rows =="
> @find $(DATA_DIR) -path '*/deduped/*.jsonl' -type f | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

pretrain-push:
> $(PYTHON) -m slm_synth.pretrain.push_hf --config $(CONFIG_FILE) $(PRETRAIN_SIGNAL_ARG)

distill-smoke:
> $(PYTHON) -m slm_synth.distillation.cli generate-seed-run \
>   $(DISTILL_SIGNALS_ARG) \
>   --count-per-signal $(DISTILL_SMOKE_COUNT_PER_SIGNAL) \
>   --output-dir $(DISTILL_OUTPUT_DIR) \
>   --manifest-dir $(DISTILL_MANIFEST_DIR) \
>   --teacher-model $(DISTILL_MODEL) \
>   --generation-run $(DISTILL_RUN) \
>   --max-tokens $(DISTILL_MAX_TOKENS) \
>   --batch-size $(DISTILL_BATCH_SIZE)
> $(MAKE) distill-report DISTILL_REPORT_RUN=$(DISTILL_RUN)

distill-generate:
> $(PYTHON) -m slm_synth.distillation.cli generate-seed-run \
>   $(DISTILL_SIGNALS_ARG) \
>   --target-preset $(DISTILL_TARGET_SIZE) \
>   --estimated-tokens-per-row $(DISTILL_ESTIMATED_TOKENS_PER_ROW) \
>   --output-dir $(DISTILL_OUTPUT_DIR) \
>   --manifest-dir $(DISTILL_MANIFEST_DIR) \
>   --teacher-model $(DISTILL_MODEL) \
>   --generation-run $(DISTILL_TARGET_RUN) \
>   --max-tokens $(DISTILL_MAX_TOKENS) \
>   --batch-size $(DISTILL_BATCH_SIZE)
> $(MAKE) distill-report DISTILL_REPORT_RUN=$(DISTILL_TARGET_RUN)

distill-report:
> $(PYTHON) -m slm_synth.distillation.cli report-coverage \
>   --run-manifest $(DISTILL_MANIFEST_DIR)/$(DISTILL_REPORT_RUN).manifest.json \
>   --output $(DISTILL_COVERAGE_REPORT)
> $(PYTHON) -m slm_synth.distillation.cli build-dataset-card \
>   --run-manifest $(DISTILL_MANIFEST_DIR)/$(DISTILL_REPORT_RUN).manifest.json \
>   --output $(DISTILL_DATASET_CARD) \
>   --dataset-name "$(DISTILL_DATASET_NAME)"

distill-inspect:
> @echo "== distillation files =="
> @find $(DISTILL_OUTPUT_DIR) $(DISTILL_MANIFEST_DIR) -type f 2>/dev/null | sort
> @echo "== distillation sample rows =="
> @find $(DISTILL_OUTPUT_DIR) -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

sft-smoke:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_SMOKE_FAMILIES) \
>   --count-per-family $(SFT_SMOKE_COUNT_PER_FAMILY) \
>   --batch-size $(SFT_SMOKE_BATCH_SIZE) \
>   --output-dir $(SFT_OUTPUT_DIR) \
>   --manifest-dir $(SFT_MANIFEST_DIR) \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --concurrency $(SFT_CONCURRENCY)
> $(MAKE) sft-report

sft-generate:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_FAMILIES) \
>   --count-per-family $(SFT_COUNT_PER_FAMILY) \
>   --batch-size $(SFT_BATCH_SIZE) \
>   --output-dir $(SFT_OUTPUT_DIR) \
>   --manifest-dir $(SFT_MANIFEST_DIR) \
>   --teacher-model $(SFT_MODEL) \
>   --generation-run $(SFT_TARGET_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --concurrency $(SFT_CONCURRENCY)
> $(MAKE) sft-report

sft-report:
> $(PYTHON) -m slm_synth.sft.cli report-coverage \
>   --input $(SFT_OUTPUT_DIR) \
>   --output $(SFT_COVERAGE_REPORT)

sft-inspect:
> @echo "== SFT files =="
> @find $(SFT_OUTPUT_DIR) $(SFT_MANIFEST_DIR) -type f 2>/dev/null | sort
> @echo "== SFT sample rows =="
> @find $(SFT_OUTPUT_DIR) -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

dpo-smoke:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --families $(DPO_SMOKE_FAMILIES) \
>   --count-per-family $(DPO_SMOKE_COUNT_PER_FAMILY) \
>   --batch-size $(DPO_SMOKE_BATCH_SIZE) \
>   --output-dir $(DPO_OUTPUT_DIR) \
>   --manifest-dir $(DPO_MANIFEST_DIR) \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --concurrency $(DPO_CONCURRENCY)
> $(MAKE) dpo-report

dpo-generate:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --families $(DPO_FAMILIES) \
>   --count-per-family $(DPO_COUNT_PER_FAMILY) \
>   --batch-size $(DPO_BATCH_SIZE) \
>   --output-dir $(DPO_OUTPUT_DIR) \
>   --manifest-dir $(DPO_MANIFEST_DIR) \
>   --teacher-model $(DPO_MODEL) \
>   --generation-run $(DPO_TARGET_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --concurrency $(DPO_CONCURRENCY)
> $(MAKE) dpo-report

dpo-report:
> $(PYTHON) -m slm_synth.dpo.cli report-coverage \
>   --input $(DPO_OUTPUT_DIR) \
>   --output $(DPO_COVERAGE_REPORT)

dpo-inspect:
> @echo "== DPO files =="
> @find $(DPO_OUTPUT_DIR) $(DPO_MANIFEST_DIR) -type f 2>/dev/null | sort
> @echo "== DPO sample rows =="
> @find $(DPO_OUTPUT_DIR) -name '*.jsonl' -type f 2>/dev/null | sort | head -n 5 | xargs -r -I{} sh -c 'echo "--- {}"; head -n 3 "{}"'

test:
> $(PYTHON) -m compileall -q slm_synth tests
> pytest -q

clean:
> rm -rf $(DATA_DIR) data/distillation data/sft data/dpo
