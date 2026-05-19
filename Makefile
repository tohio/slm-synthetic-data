# ============================================================
# SLM Synthetic Data Pipeline — Makefile
# ============================================================

.RECIPEPREFIX := >

# -----------------------------
# Global Variables
# -----------------------------
PYTHON := python
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

PROFILE ?= balanced
TOKENS ?= 200000
MODEL ?=
BATCH ?=
CONCURRENCY ?=
SERVICE_TIER ?= flex
CONFIG_FILE ?= configs/synthetic.yaml

DATA_DIR ?= data
HF_USERNAME ?=
HF_REPO ?=

SIGNAL ?=
SIGNAL_ARG := $(if $(SIGNAL),--signal $(SIGNAL),)

.PHONY: help configure list-models generate validate dedup push all test clean

# -----------------------------
# Help
# -----------------------------
help:
> @echo ""
> @echo "SLM Synthetic Data Pipeline"
> @echo "==========================="
> @echo ""
> @echo "Usage: make <target> [PROFILE=name] [TOKENS=N] [MODEL=name] [SIGNAL=name] [BATCH=N] [CONCURRENCY=N]"
> @echo ""
> @echo "Configuration:"
> @echo "  configure              Generate configs/synthetic.yaml using profile presets"
> @echo "  list-models            List available Groq models"
> @echo ""
> @echo "Pipeline:"
> @echo "  generate               Run synthetic data generation"
> @echo "  validate               Validate generated samples"
> @echo "  dedup                  Deduplicate dataset"
> @echo "  push                   Push dataset to Hugging Face"
> @echo "  all                    Run generate -> validate -> dedup -> push"
> @echo ""
> @echo "Testing:"
> @echo "  test                   Run all tests"
> @echo ""
> @echo "Utilities:"
> @echo "  clean                  Remove generated data"
> @echo ""
> @echo "Examples:"
> @echo "  make configure PROFILE=balanced TOKENS=200000"
> @echo "  make configure PROFILE=speed TOKENS=600000000"
> @echo "  make configure PROFILE=quality TOKENS=50000000"
> @echo "  make generate SIGNAL=arithmetic"
> @echo ""

# -----------------------------
# Configuration Helper
# -----------------------------
configure:
> $(PYTHON) configs/configure_synthetic.py \
>   --profile "$(PROFILE)" \
>   --tokens $(TOKENS) \
>   --service-tier "$(SERVICE_TIER)" \
>   $(if $(MODEL),--model $(MODEL),) \
>   $(if $(BATCH),--batch-size $(BATCH),) \
>   $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),)

# -----------------------------
# List available Groq models
# -----------------------------
list-models:
> @echo "Fetching available Groq models..."
> @$(PYTHON) -c 'import os, requests; from dotenv import load_dotenv; load_dotenv(); api_key = os.getenv("GROQ_API_KEY"); import sys; \
if not api_key: print("ERROR: GROQ_API_KEY not found in .env"); sys.exit(1); \
resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}); \
resp.raise_for_status(); models = sorted(m["id"] for m in resp.json().get("data", [])); \
print("\nAvailable Groq Models:"); [print(" -", m) for m in models]; print()'

# -----------------------------
# Pipeline Targets
# -----------------------------
generate:
> $(PYTHON) -m slm_synth.generate --config $(CONFIG_FILE) $(SIGNAL_ARG)

validate:
> $(PYTHON) -m slm_synth.validate --config $(CONFIG_FILE)

dedup:
> $(PYTHON) -m slm_synth.dedup --config $(CONFIG_FILE)

push:
	python -m slm_synth.push_hf --config configs/synthetic.yaml $(if $(SIGNAL),--signal $(SIGNAL),)
> $(PYTHON) -m slm_synth.push_hf --config $(CONFIG_FILE)

all: generate validate dedup push

# -----------------------------
# Testing
# -----------------------------
test:
> pytest -q

# -----------------------------
# Utilities
# -----------------------------
clean:
> rm -rf $(DATA_DIR)/synthetic
