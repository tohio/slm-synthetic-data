# ============================================================
# SLM Synthetic Data Pipeline — Makefile
# ============================================================

.RECIPEPREFIX := >

PYTHON := python
PROFILE ?= balanced
TOKENS ?= 200000
MODEL ?=
BATCH ?= 32
CONCURRENCY ?=
MAX_TOKENS ?=
RUN ?=
HF_REPO ?=
CONFIG_FILE ?= configs/synthetic.yaml
STAGE ?= deduped
DATA_DIR ?= data/runs
SIGNAL ?=
SIGNAL_ARG := $(if $(SIGNAL),--signal $(SIGNAL),)
PRETRAIN_MANIFEST ?=
PRETRAIN_MANIFEST_ARG := $(if $(PRETRAIN_MANIFEST),--output $(PRETRAIN_MANIFEST),)
PRETRAIN_COVERAGE_REPORT ?=
PRETRAIN_COVERAGE_REPORT_ARG := $(if $(PRETRAIN_COVERAGE_REPORT),--output $(PRETRAIN_COVERAGE_REPORT),)
PRETRAIN_GENERATION_RUN ?=
PRETRAIN_GENERATION_RUN_ARG := $(if $(PRETRAIN_GENERATION_RUN),--generation-run $(PRETRAIN_GENERATION_RUN),)
DISTILL_SIGNAL ?= arithmetic
DISTILL_SIGNALS ?=
DISTILL_COUNT ?= 2
DISTILL_START_INDEX ?= 1
DISTILL_TARGET ?= smoke
DISTILL_ESTIMATED_TOKENS_PER_ROW ?= 512
DISTILL_PROMPTS ?= /tmp/$(DISTILL_SIGNAL).prompts.jsonl
DISTILL_TEACHER_PROMPT ?= /tmp/$(DISTILL_SIGNAL).teacher_prompt.txt
DISTILL_TEACHER_RESPONSE ?= /tmp/$(DISTILL_SIGNAL).teacher_response.json
DISTILL_OUTPUT_DIR ?= data/distillation/datasets
DISTILL_MANIFEST_DIR ?= data/distillation/manifests
DISTILL_TEACHER_MODEL ?=
DISTILL_GENERATION_RUN ?= smoke-001
DISTILL_MAX_TOKENS ?= 1024
DISTILL_TOKEN_TARGET ?=
DISTILL_DATASET_CARD ?= data/distillation/README.md
DISTILL_DATASET_NAME ?= SLM Synthetic Distillation
DISTILL_LICENSE ?=
DISTILL_RUN_MANIFEST ?= $(DISTILL_MANIFEST_DIR)/$(DISTILL_GENERATION_RUN).manifest.json
DISTILL_COVERAGE_REPORT ?= data/distillation/coverage.json
DISTILL_SIGNALS_ARG := $(if $(DISTILL_SIGNALS),--signals $(DISTILL_SIGNALS),)
DISTILL_TOKEN_TARGET_ARG := $(if $(DISTILL_TOKEN_TARGET),--token-target $(DISTILL_TOKEN_TARGET),)
DISTILL_LICENSE_ARG := $(if $(DISTILL_LICENSE),--license $(DISTILL_LICENSE),)
SFT_FAMILY ?= answer_only_arithmetic
SFT_SPEC_FAMILY ?= basic_arithmetic_qa
SFT_FAMILIES ?= all
SFT_COUNT ?= 100
SFT_COUNT_PER_FAMILY ?= $(SFT_COUNT)
SFT_OUTPUT_DIR ?= data/sft/datasets
SFT_MANIFEST_DIR ?= data/sft/manifests
SFT_COVERAGE_REPORT ?= data/sft/coverage.json
SFT_GENERATION_RUN ?= sft-smoke-001
SFT_START_INDEX ?= 1
SFT_SPECS ?= /tmp/sft.specs.jsonl
SFT_TEACHER_RESPONSE ?= /tmp/sft.teacher_response.json
SFT_OUTPUT ?= $(SFT_OUTPUT_DIR)/$(SFT_SPEC_FAMILY).jsonl
SFT_MANIFEST ?= $(SFT_MANIFEST_DIR)/$(SFT_SPEC_FAMILY).$(SFT_GENERATION_RUN).manifest.json
SFT_TEACHER_MODEL ?=
SFT_MAX_TOKENS ?= 4096
SFT_BATCH_SIZE ?= 5
SFT_MAX_WORKERS ?= 1
SFT_RUN_MANIFEST_FILENAME ?=
SFT_RUN_MANIFEST_ARG := $(if $(SFT_RUN_MANIFEST_FILENAME),--run-manifest-filename $(SFT_RUN_MANIFEST_FILENAME),)
DPO_FAMILY ?= answer_only_arithmetic
DPO_SPEC_FAMILY ?= basic_arithmetic_qa
DPO_FAMILIES ?= all
DPO_COUNT ?= 100
DPO_COUNT_PER_FAMILY ?= $(DPO_COUNT)
DPO_OUTPUT_DIR ?= data/dpo/datasets
DPO_MANIFEST_DIR ?= data/dpo/manifests
DPO_COVERAGE_REPORT ?= data/dpo/coverage.json
DPO_GENERATION_RUN ?= dpo-smoke-001
DPO_START_INDEX ?= 1
DPO_SPECS ?= /tmp/dpo.specs.jsonl
DPO_TEACHER_RESPONSE ?= /tmp/dpo.teacher_response.json
DPO_OUTPUT ?= $(DPO_OUTPUT_DIR)/$(DPO_SPEC_FAMILY).jsonl
DPO_MANIFEST ?= $(DPO_MANIFEST_DIR)/$(DPO_SPEC_FAMILY).$(DPO_GENERATION_RUN).manifest.json
DPO_TEACHER_MODEL ?=
DPO_MAX_TOKENS ?= 4096
DPO_BATCH_SIZE ?= 5
DPO_MAX_WORKERS ?= 1
DPO_RUN_MANIFEST_FILENAME ?=
DPO_RUN_MANIFEST_ARG := $(if $(DPO_RUN_MANIFEST_FILENAME),--run-manifest-filename $(DPO_RUN_MANIFEST_FILENAME),)

.PHONY: help configure production-config preflight-artifacts generate validate dedup pretrain-generate-manifest pretrain-report-coverage report-duplicates report-artifacts report-lengths push all distill-plan distill-build-prompts distill-render-teacher-prompt distill-materialize-batch distill-generate-batch distill-generate-seed-run distill-build-dataset-card distill-report-coverage sft-build-specs sft-materialize-llm-batch sft-generate-llm-batch sft-generate-llm-run sft-materialize-seed sft-materialize-seed-run sft-report-coverage dpo-build-specs dpo-materialize-llm-batch dpo-generate-llm-batch dpo-generate-llm-run dpo-materialize-seed dpo-materialize-seed-run dpo-report-coverage test clean

help:
> @echo ""
> @echo "SLM Synthetic Data Pipeline — Grounded OpenRouter Generation"
> @echo "==========================================================="
> @echo ""
> @echo "Usage: make <target> [PROFILE=name] [TOKENS=N] [MODEL=name] [SIGNAL=name] [BATCH=N] [CONCURRENCY=N] [MAX_TOKENS=N] [RUN=name]"
> @echo ""
> @echo "Configuration:"
> @echo "  configure              Write configs/synthetic.yaml for a grounded run"
> @echo "  production-config      Configure the locked 762.5M estimated-token corpus run"
> @echo ""
> @echo "Pipeline:"
> @echo "  preflight-artifacts    Validate all planned grounded artifacts before paid rendering"
> @echo "  generate               Generate grounded rendered records through OpenRouter/DeepSeek"
> @echo "  validate               Validate raw JSONL records"
> @echo "  dedup                  Exact-deduplicate validated records"
> @echo "  pretrain-generate-manifest Generate a pretrain run-level manifest"
> @echo "  pretrain-report-coverage Generate a pretrain coverage report"
> @echo "  report-artifacts       Report grounded artifact duplicates/family coverage"
> @echo "  report-duplicates      Report exact duplicates in rendered records"
> @echo "  report-lengths         Estimate record length for avg_tokens_per_sample calibration"
> @echo "  push                   Push deduped data to Hugging Face"
> @echo "  all                    Generate -> reports -> validate -> dedup -> reports"
> @echo ""
> @echo "Distillation:"
> @echo "  distill-plan                 Print approximate row counts for a token target"
> @echo "  distill-build-prompts        Build seed prompt JSONL for one signal"
> @echo "  distill-render-teacher-prompt Render a local teacher batch prompt"
> @echo "  distill-materialize-batch    Merge a local teacher response into JSONL + manifest"
> @echo "  distill-generate-batch       Generate one signal batch through OpenRouter"
> @echo "  distill-generate-seed-run    Generate seed batches across one or more signals"
> @echo "  distill-build-dataset-card   Build a dataset card from a run manifest"
> @echo "  distill-report-coverage      Write distillation coverage report JSON"
> @echo ""
> @echo "SFT/DPO:"
> @echo "  sft-build-specs             Build SFT LLM task spec JSONL"
> @echo "  sft-materialize-llm-batch   Merge saved SFT teacher JSON into JSONL + manifest"
> @echo "  sft-generate-llm-batch      Generate one SFT batch through OpenRouter"
> @echo "  sft-generate-llm-run        Generate SFT datasets across families and batches"
> @echo "  sft-materialize-seed         Build deterministic SFT seed JSONL + manifest"
> @echo "  sft-materialize-seed-run     Build deterministic SFT seed datasets across families"
> @echo "  sft-report-coverage          Write SFT coverage report JSON"
> @echo "  dpo-build-specs             Build DPO LLM task spec JSONL"
> @echo "  dpo-materialize-llm-batch   Merge saved DPO teacher JSON into JSONL + manifest"
> @echo "  dpo-generate-llm-batch      Generate one DPO batch through OpenRouter"
> @echo "  dpo-generate-llm-run        Generate DPO datasets across families and batches"
> @echo "  dpo-materialize-seed         Build deterministic DPO seed JSONL + manifest"
> @echo "  dpo-materialize-seed-run     Build deterministic DPO seed datasets across families"
> @echo "  dpo-report-coverage          Write DPO coverage report JSON"
> @echo ""
> @echo "Examples:"
> @echo "  make configure TOKENS=250000 BATCH=64 CONCURRENCY=8 MAX_TOKENS=16384 RUN=batch_qual_250k_b64_c8"
> @echo "  make generate"
> @echo "  make pretrain-generate-manifest"
> @echo "  make pretrain-report-coverage"
> @echo "  make report-artifacts"
> @echo "  make production-config CONCURRENCY=4"
> @echo "  make distill-plan DISTILL_TARGET=smoke"
> @echo "  make distill-generate-seed-run DISTILL_TEACHER_MODEL=openai/gpt-4.1-mini DISTILL_TARGET=smoke"
> @echo "  make distill-report-coverage"
> @echo "  make sft-build-specs SFT_SPEC_FAMILY=basic_arithmetic_qa SFT_COUNT=25"
> @echo "  make sft-generate-llm-run SFT_FAMILIES='basic_arithmetic_qa repeat_exact_n_times' SFT_COUNT_PER_FAMILY=25 SFT_BATCH_SIZE=5 SFT_TEACHER_MODEL=openai/gpt-4.1-mini"
> @echo "  make sft-materialize-seed SFT_FAMILY=repeat_exact_n_times SFT_COUNT=25"
> @echo "  make sft-materialize-seed-run SFT_FAMILIES='answer_only_arithmetic repeat_exact_n_times' SFT_COUNT_PER_FAMILY=25"
> @echo "  make sft-report-coverage"
> @echo "  make dpo-build-specs DPO_SPEC_FAMILY=basic_arithmetic_qa DPO_COUNT=25"
> @echo "  make dpo-generate-llm-run DPO_FAMILIES='basic_arithmetic_qa repeat_exact_n_times' DPO_COUNT_PER_FAMILY=25 DPO_BATCH_SIZE=5 DPO_TEACHER_MODEL=openai/gpt-4.1-mini"
> @echo "  make dpo-materialize-seed DPO_FAMILY=repeat_exact_n_times DPO_COUNT=25"
> @echo "  make dpo-materialize-seed-run DPO_FAMILIES='answer_only_arithmetic repeat_exact_n_times' DPO_COUNT_PER_FAMILY=25"
> @echo "  make dpo-report-coverage"
> @echo ""

configure:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(TOKENS) \
>   --batch-size $(BATCH) \
>   $(if $(MODEL),--model $(MODEL),) \
>   $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) \
>   $(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS),) \
>   $(if $(RUN),--run $(RUN),) \
>   $(if $(HF_REPO),--hf_repo $(HF_REPO),)

production-config:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens 762500000 \
>   --run grounded_production_762_5m \
>   --batch-size 32 \
>   $(if $(MODEL),--model $(MODEL),) \
>   $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),)

preflight-artifacts:
> $(PYTHON) -m slm_synth.preflight_artifacts --config $(CONFIG_FILE) $(SIGNAL_ARG)

generate:
> $(PYTHON) -m slm_synth.generate --config $(CONFIG_FILE) $(SIGNAL_ARG)

validate:
> $(PYTHON) -m slm_synth.validate --config $(CONFIG_FILE) $(SIGNAL_ARG)

dedup:
> $(PYTHON) -m slm_synth.dedup --config $(CONFIG_FILE) $(SIGNAL_ARG)

pretrain-generate-manifest:
> $(PYTHON) -m slm_synth.pretrain.manifest \
>   --config $(CONFIG_FILE) \
>   $(PRETRAIN_MANIFEST_ARG) \
>   $(PRETRAIN_GENERATION_RUN_ARG)

pretrain-report-coverage:
> $(PYTHON) -m slm_synth.pretrain.manifest \
>   --config $(CONFIG_FILE) \
>   $(PRETRAIN_COVERAGE_REPORT_ARG) \
>   $(PRETRAIN_GENERATION_RUN_ARG)

report-artifacts:
> $(PYTHON) -m slm_synth.report_artifacts --config $(CONFIG_FILE) $(SIGNAL_ARG)

report-duplicates:
> $(PYTHON) -m slm_synth.report_duplicates --config $(CONFIG_FILE) --stage $(STAGE)

report-lengths:
> $(PYTHON) -m slm_synth.report_lengths --config $(CONFIG_FILE) --stage $(STAGE)

push:
> $(PYTHON) -m slm_synth.push_hf --config $(CONFIG_FILE) $(SIGNAL_ARG)

all: preflight-artifacts generate report-artifacts validate dedup report-duplicates report-lengths

distill-plan:
> $(PYTHON) -m slm_synth.distillation.cli plan-token-target \
>   --target $(DISTILL_TARGET) \
>   $(DISTILL_SIGNALS_ARG) \
>   --estimated-tokens-per-row $(DISTILL_ESTIMATED_TOKENS_PER_ROW)

distill-build-prompts:
> $(PYTHON) -m slm_synth.distillation.cli build-seed-prompts \
>   --signal $(DISTILL_SIGNAL) \
>   --count $(DISTILL_COUNT) \
>   --start-index $(DISTILL_START_INDEX) \
>   --output $(DISTILL_PROMPTS)

distill-render-teacher-prompt:
> $(PYTHON) -m slm_synth.distillation.cli render-teacher-prompt \
>   --signal $(DISTILL_SIGNAL) \
>   --prompts $(DISTILL_PROMPTS) \
>   --output $(DISTILL_TEACHER_PROMPT)

distill-materialize-batch:
> $(PYTHON) -m slm_synth.distillation.cli materialize-batch \
>   --signal $(DISTILL_SIGNAL) \
>   --prompts $(DISTILL_PROMPTS) \
>   --teacher-response $(DISTILL_TEACHER_RESPONSE) \
>   --output-dir $(DISTILL_OUTPUT_DIR) \
>   --manifest-dir $(DISTILL_MANIFEST_DIR) \
>   --teacher-model $(DISTILL_TEACHER_MODEL) \
>   --generation-run $(DISTILL_GENERATION_RUN) \
>   $(DISTILL_TOKEN_TARGET_ARG)

distill-generate-batch:
> $(PYTHON) -m slm_synth.distillation.cli generate-batch \
>   --signal $(DISTILL_SIGNAL) \
>   --prompts $(DISTILL_PROMPTS) \
>   --output-dir $(DISTILL_OUTPUT_DIR) \
>   --manifest-dir $(DISTILL_MANIFEST_DIR) \
>   --teacher-model $(DISTILL_TEACHER_MODEL) \
>   --generation-run $(DISTILL_GENERATION_RUN) \
>   --max-tokens $(DISTILL_MAX_TOKENS) \
>   $(DISTILL_TOKEN_TARGET_ARG)

distill-generate-seed-run:
> $(PYTHON) -m slm_synth.distillation.cli generate-seed-run \
>   $(DISTILL_SIGNALS_ARG) \
>   --target-preset $(DISTILL_TARGET) \
>   --estimated-tokens-per-row $(DISTILL_ESTIMATED_TOKENS_PER_ROW) \
>   --output-dir $(DISTILL_OUTPUT_DIR) \
>   --manifest-dir $(DISTILL_MANIFEST_DIR) \
>   --teacher-model $(DISTILL_TEACHER_MODEL) \
>   --generation-run $(DISTILL_GENERATION_RUN) \
>   --max-tokens $(DISTILL_MAX_TOKENS)

distill-build-dataset-card:
> $(PYTHON) -m slm_synth.distillation.cli build-dataset-card \
>   --run-manifest $(DISTILL_RUN_MANIFEST) \
>   --output $(DISTILL_DATASET_CARD) \
>   --dataset-name "$(DISTILL_DATASET_NAME)" \
>   $(DISTILL_LICENSE_ARG)

distill-report-coverage:
> $(PYTHON) -m slm_synth.distillation.cli report-coverage \
>   --run-manifest $(DISTILL_RUN_MANIFEST) \
>   --output $(DISTILL_COVERAGE_REPORT)

sft-build-specs:
> $(PYTHON) -m slm_synth.sft.cli build-specs \
>   --family $(SFT_SPEC_FAMILY) \
>   --count $(SFT_COUNT) \
>   --output $(SFT_SPECS) \
>   --start-index $(SFT_START_INDEX)

sft-materialize-llm-batch:
> $(PYTHON) -m slm_synth.sft.cli materialize-llm-batch \
>   --specs $(SFT_SPECS) \
>   --teacher-response $(SFT_TEACHER_RESPONSE) \
>   --output $(SFT_OUTPUT) \
>   --manifest $(SFT_MANIFEST) \
>   --teacher-model $(SFT_TEACHER_MODEL) \
>   --generation-run $(SFT_GENERATION_RUN)

sft-generate-llm-batch:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-batch \
>   --specs $(SFT_SPECS) \
>   --output $(SFT_OUTPUT) \
>   --manifest $(SFT_MANIFEST) \
>   --teacher-model $(SFT_TEACHER_MODEL) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS)

sft-generate-llm-run:
> $(PYTHON) -m slm_synth.sft.cli generate-llm-run \
>   --families $(SFT_FAMILIES) \
>   --count-per-family $(SFT_COUNT_PER_FAMILY) \
>   --batch-size $(SFT_BATCH_SIZE) \
>   --output-dir $(SFT_OUTPUT_DIR) \
>   --manifest-dir $(SFT_MANIFEST_DIR) \
>   --teacher-model $(SFT_TEACHER_MODEL) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --max-tokens $(SFT_MAX_TOKENS) \
>   --start-index $(SFT_START_INDEX) \
>   --max-workers $(SFT_MAX_WORKERS) \
>   $(SFT_RUN_MANIFEST_ARG)

sft-materialize-seed:
> $(PYTHON) -m slm_synth.sft.cli materialize-seed-dataset \
>   --family $(SFT_FAMILY) \
>   --count $(SFT_COUNT) \
>   --output-dir $(SFT_OUTPUT_DIR) \
>   --manifest-dir $(SFT_MANIFEST_DIR) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --start-index $(SFT_START_INDEX)

sft-materialize-seed-run:
> $(PYTHON) -m slm_synth.sft.cli materialize-seed-run \
>   --families $(SFT_FAMILIES) \
>   --count-per-family $(SFT_COUNT_PER_FAMILY) \
>   --output-dir $(SFT_OUTPUT_DIR) \
>   --manifest-dir $(SFT_MANIFEST_DIR) \
>   --generation-run $(SFT_GENERATION_RUN) \
>   --start-index $(SFT_START_INDEX) \
>   $(SFT_RUN_MANIFEST_ARG)

sft-report-coverage:
> $(PYTHON) -m slm_synth.sft.cli report-coverage \
>   --input $(SFT_OUTPUT_DIR) \
>   --output $(SFT_COVERAGE_REPORT)

dpo-build-specs:
> $(PYTHON) -m slm_synth.dpo.cli build-specs \
>   --family $(DPO_SPEC_FAMILY) \
>   --count $(DPO_COUNT) \
>   --output $(DPO_SPECS) \
>   --start-index $(DPO_START_INDEX)

dpo-materialize-llm-batch:
> $(PYTHON) -m slm_synth.dpo.cli materialize-llm-batch \
>   --specs $(DPO_SPECS) \
>   --teacher-response $(DPO_TEACHER_RESPONSE) \
>   --output $(DPO_OUTPUT) \
>   --manifest $(DPO_MANIFEST) \
>   --teacher-model $(DPO_TEACHER_MODEL) \
>   --generation-run $(DPO_GENERATION_RUN)

dpo-generate-llm-batch:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-batch \
>   --specs $(DPO_SPECS) \
>   --output $(DPO_OUTPUT) \
>   --manifest $(DPO_MANIFEST) \
>   --teacher-model $(DPO_TEACHER_MODEL) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS)

dpo-generate-llm-run:
> $(PYTHON) -m slm_synth.dpo.cli generate-llm-run \
>   --families $(DPO_FAMILIES) \
>   --count-per-family $(DPO_COUNT_PER_FAMILY) \
>   --batch-size $(DPO_BATCH_SIZE) \
>   --output-dir $(DPO_OUTPUT_DIR) \
>   --manifest-dir $(DPO_MANIFEST_DIR) \
>   --teacher-model $(DPO_TEACHER_MODEL) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --max-tokens $(DPO_MAX_TOKENS) \
>   --start-index $(DPO_START_INDEX) \
>   --max-workers $(DPO_MAX_WORKERS) \
>   $(DPO_RUN_MANIFEST_ARG)

dpo-materialize-seed:
> $(PYTHON) -m slm_synth.dpo.cli materialize-seed-dataset \
>   --family $(DPO_FAMILY) \
>   --count $(DPO_COUNT) \
>   --output-dir $(DPO_OUTPUT_DIR) \
>   --manifest-dir $(DPO_MANIFEST_DIR) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --start-index $(DPO_START_INDEX)

dpo-materialize-seed-run:
> $(PYTHON) -m slm_synth.dpo.cli materialize-seed-run \
>   --families $(DPO_FAMILIES) \
>   --count-per-family $(DPO_COUNT_PER_FAMILY) \
>   --output-dir $(DPO_OUTPUT_DIR) \
>   --manifest-dir $(DPO_MANIFEST_DIR) \
>   --generation-run $(DPO_GENERATION_RUN) \
>   --start-index $(DPO_START_INDEX) \
>   $(DPO_RUN_MANIFEST_ARG)

dpo-report-coverage:
> $(PYTHON) -m slm_synth.dpo.cli report-coverage \
>   --input $(DPO_OUTPUT_DIR) \
>   --output $(DPO_COVERAGE_REPORT)

test:
> pytest -q

clean:
> rm -rf $(DATA_DIR)
