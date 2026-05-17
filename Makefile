# ============================================================
# SLM Synthetic Data Pipeline — Makefile
# ============================================================

# -----------------------------
# Global Variables
# -----------------------------
PYTHON := python
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

MODEL ?= openai/gpt-oss-20b
TOKENS ?= 200000
BATCH ?=
CONFIG_FILE ?= configs/synthetic.yaml

DATA_DIR ?= data
HF_USERNAME ?=
HF_REPO ?=

# -----------------------------
# Help
# -----------------------------
help:
    @echo ""
    @echo "SLM Synthetic Data Pipeline"
    @echo "==========================="
    @echo ""
    @echo "Usage: make <target> [MODEL=name] [TOKENS=N] [BATCH=N]"
    @echo ""
    @echo "Configuration:"
    @echo "  configure              Update configs/synthetic.yaml using Groq model profiles"
    @echo "  list-models            List available Groq models"
    @echo ""
    @echo "Pipeline:"
    @echo "  generate               Run synthetic data generation"
    @echo "  validate               Validate generated samples"
    @echo "  dedup                  Deduplicate dataset"
    @echo "  push                   Push dataset to Hugging Face"
    @echo ""
    @echo "Testing:"
    @echo "  test                   Run all tests"
    @echo ""
    @echo "Utilities:"
    @echo "  clean                  Remove generated data"
    @echo ""

# -----------------------------
# Configuration Helper
# -----------------------------
configure:
ifeq ($(BATCH),)
    $(PYTHON) configs/configure_synthetic.py \
        --model "$(MODEL)" \
        --tokens $(TOKENS) \
        --config $(CONFIG_FILE)
else
    $(PYTHON) configs/configure_synthetic.py \
        --model "$(MODEL)" \
        --tokens $(TOKENS) \
        --batch-size $(BATCH) \
        --config $(CONFIG_FILE)
endif

# -----------------------------
# List available Groq models
# -----------------------------
list-models:
    @echo "Fetching available Groq models..."
    @$(PYTHON) - << 'EOF'
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in environment or .env file.")
    raise SystemExit(1)

url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

resp = requests.get(url, headers=headers)
resp.raise_for_status()

data = resp.json()
models = [m["id"] for m in data.get("data", [])]

print("\nAvailable Groq Models:")
for m in models:
    print(" -", m)
print()
EOF

# -----------------------------
# Pipeline Targets
# -----------------------------
generate:
    $(PYTHON) generate.py --config $(CONFIG_FILE)

validate:
    $(PYTHON) validate.py --config $(CONFIG_FILE)

dedup:
    $(PYTHON) dedup.py --config $(CONFIG_FILE)

push:
    $(PYTHON) push_to_hf.py --config $(CONFIG_FILE)

# -----------------------------
# Testing
# -----------------------------
test:
    pytest -q

# -----------------------------
# Utilities
# -----------------------------
clean:
    rm -rf $(DATA_DIR)/synthetic
