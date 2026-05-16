# ------------------------------------------------------------
# SLM Synthetic Data — Makefile
# ------------------------------------------------------------

ENV_FILE := .env
CONFIG := configs/synthetic.yaml

# Load environment variables
include $(ENV_FILE)
export $(shell sed 's/=.*//' $(ENV_FILE))

# ------------------------------------------------------------
# Core Pipeline Commands
# ------------------------------------------------------------

generate:
    python -m slm_synth.generate $(CONFIG)

validate:
    python -m slm_synth.validate data/runs/synthetic/raw

dedup:
    python -m slm_synth.dedup data/runs/synthetic/validated

push:
    python -m slm_synth.push_hf data/runs/synthetic/deduped

# ------------------------------------------------------------
# Full Pipeline
# ------------------------------------------------------------

all: generate validate dedup push

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

test:
    pytest -q

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

clean:
    rm -rf data/runs/synthetic

# ── Help ──────────────────────────────────────────────────────────────────────

help:
    @echo ""
    @echo "SLM Synthetic Data Pipeline"
    @echo "==========================="
    @echo ""
    @echo "Usage: make <target>"
    @echo ""
    @echo "Core Pipeline:"
    @echo "  [generate](ca://s?q=Run_synthetic_data_generation)        Run LLM sampling and write raw JSONL outputs"
    @echo "  [validate](ca://s?q=Run_synthetic_data_validation)        Validate raw samples against schemas"
    @echo "  [dedup](ca://s?q=Run_synthetic_data_deduplication)        Deduplicate validated samples (exact + MinHash)"
    @echo "  [push](ca://s?q=Push_synthetic_dataset_to_HuggingFace)    Upload deduped dataset to Hugging Face"
    @echo "  [all](ca://s?q=Run_full_synthetic_pipeline)               Run generate → validate → dedup → push"
    @echo ""
    @echo "Testing:"
    @echo "  [test](ca://s?q=Run_pytest_for_slm_synth)                 Run all unit tests"
    @echo ""
    @echo "Utility:"
    @echo "  [clean](ca://s?q=Clean_synthetic_run_directory)           Remove the current run directory"
    @echo ""
    @echo "Config:"
    @echo "  CONFIG=$(CONFIG)"
    @echo "  DATA_DIR=$(DATA_DIR)"
    @echo "  RUN_NAME=$(RUN_NAME)"
    @echo ""
