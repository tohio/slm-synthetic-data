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

.PHONY: help configure production-config preflight-artifacts generate validate dedup report-duplicates report-artifacts report-lengths push all test clean

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
> @echo "  report-artifacts       Report grounded artifact duplicates/family coverage"
> @echo "  report-duplicates      Report exact duplicates in rendered records"
> @echo "  report-lengths         Estimate record length for avg_tokens_per_sample calibration"
> @echo "  push                   Push deduped data to Hugging Face"
> @echo "  all                    Generate -> reports -> validate -> dedup -> reports"
> @echo ""
> @echo "Examples:"
> @echo "  make configure TOKENS=250000 BATCH=64 CONCURRENCY=8 MAX_TOKENS=16384 RUN=batch_qual_250k_b64_c8"
> @echo "  make generate"
> @echo "  make report-artifacts"
> @echo "  make production-config CONCURRENCY=4"
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

report-artifacts:
> $(PYTHON) -m slm_synth.report_artifacts --config $(CONFIG_FILE) $(SIGNAL_ARG)

report-duplicates:
> $(PYTHON) -m slm_synth.report_duplicates --config $(CONFIG_FILE) --stage $(STAGE)

report-lengths:
> $(PYTHON) -m slm_synth.report_lengths --config $(CONFIG_FILE) --stage $(STAGE)

push:
> $(PYTHON) -m slm_synth.push_hf --config $(CONFIG_FILE) $(SIGNAL_ARG)

all: preflight-artifacts generate report-artifacts validate dedup report-duplicates report-lengths

test:
> pytest -q

clean:
> rm -rf $(DATA_DIR)
