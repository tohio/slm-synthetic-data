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

MODEL ?= openai/gpt-oss-20b
TOKENS ?= 200000
BATCH ?=
SIGNAL ?=
CONFIG_FILE ?= configs/synthetic.yaml

DATA_DIR ?= data
HF_USERNAME ?=
HF_REPO ?=

SIGNAL_ARG := $(if $(SIGNAL),--signal $(SIGNAL),)
BATCH_ARG := $(if $(BATCH),--batch-size $(BATCH),)

.PHONY: help configure list-models generate validate dedup push all test clean

# -----------------------------
# Help
# -----------------------------
help:
> @echo ""
> @echo "SLM Synthetic Data Pipeline"
> @echo "==========================="
> @echo ""
> @echo "Usage: make <target> [MODEL=name] [TOKENS=N] [BATCH=N] [SIGNAL=name]"
> @echo ""
> @echo "Configuration:"
> @echo "  configure              Update configs/synthetic.yaml using Groq model profiles"
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
> @echo "  make configure TOKENS=200000"
> @echo "  make configure MODEL=llama-3.3-70b-versatile TOKENS=300000000"
> @echo "  make configure MODEL=llama-3.1-8b-instant TOKENS=200000 BATCH=8"
> @echo "  make generate SIGNAL=arithmetic"
> @echo ""

# -----------------------------
# Configuration Helper
# -----------------------------
configure:
> $(PYTHON) configs/configure_synthetic.py \
>   --model "$(MODEL)" \
>   --tokens $(TOKENS) \
>   --config $(CONFIG_FILE) \
>   $(BATCH_ARG)

# -----------------------------
# List available Groq models
# -----------------------------
list-models:
> @echo "Fetching available Groq models..."
> @$(PYTHON) -c 'import os, requests; from dotenv import load_dotenv; load_dotenv(); api_key = os.environ.get("GROQ_API_KEY"); import sys; print("ERROR: GROQ_API_KEY not found in environment or .env file.") or sys.exit(1) if not api_key else None; resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30); resp.raise_for_status(); models = sorted(m["id"] for m in resp.json().get("data", []) if "id" in m); print("\nAvailable Groq Models:"); [print(" -", m) for m in models]; print()'

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
