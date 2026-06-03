# ============================================================
# SLM Distillation — Makefile
# ============================================================

.RECIPEPREFIX := >

PYTHON := python3
PYTHONPATH := .
CONFIG ?= configs/response_distill.yaml
TEACHERS_CONFIG ?= configs/teachers.yaml
LIMIT ?=

.PHONY: help install test test-unit generate generate-dry-run validate dataset response-pipeline response-pipeline-dry-run clean-generated

help:
> @echo ""
> @echo "SLM Distillation"
> @echo "================"
> @echo ""
> @echo "Usage: make <target> [CONFIG=path] [TEACHERS_CONFIG=path] [LIMIT=N]"
> @echo ""
> @echo "Setup:"
> @echo "  install                    Install Python dependencies"
> @echo ""
> @echo "Pipeline:"
> @echo "  generate-dry-run            Generate local dry-run teacher responses"
> @echo "  generate                    Generate teacher responses through configured provider"
> @echo "  validate                    Validate raw teacher responses"
> @echo "  dataset                     Build response-distillation dataset"
> @echo "  response-pipeline-dry-run   Dry-run generate -> validate -> dataset"
> @echo "  response-pipeline           Generate -> validate -> dataset"
> @echo ""
> @echo "Tests:"
> @echo "  test                        Run full test suite"
> @echo "  test-unit                   Run unit tests"
> @echo ""
> @echo "Cleanup:"
> @echo "  clean-generated             Remove generated JSONL files and runs"
> @echo ""

install:
> $(PYTHON) -m pip install -r requirements.txt

test:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

test-unit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests

generate:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_teacher_responses.py \
>   --config $(CONFIG) \
>   --teachers $(TEACHERS_CONFIG) \
>   $(if $(LIMIT),--limit $(LIMIT),)

generate-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_teacher_responses.py \
>   --config $(CONFIG) \
>   --teachers $(TEACHERS_CONFIG) \
>   $(if $(LIMIT),--limit $(LIMIT),) \
>   --dry-run

validate:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_teacher_responses.py \
>   --config $(CONFIG)

dataset:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_dataset.py \
>   --config $(CONFIG)

response-pipeline: generate validate dataset

response-pipeline-dry-run: generate-dry-run validate dataset

clean-generated:
> rm -f data/raw_teacher/*.jsonl
> rm -f data/validated/*.jsonl
> rm -f data/rejected/*.jsonl
> rm -f data/distill/*.jsonl
> rm -rf runs/