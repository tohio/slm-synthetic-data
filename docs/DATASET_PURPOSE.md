# Dataset Purpose

This repository generates synthetic data signals that can be mixed into the training corpus for small language model pretraining and continued-pretraining.

The generated signals are designed to supplement broader training data with targeted coverage for arithmetic, Python code, educational QA, and factual-restraint behavior.

## Signal Scope

- `arithmetic` — synthetic arithmetic signal for numeric reasoning coverage.
- `task_code` — synthetic Python code signal for code-pattern exposure.
- `educational_qa_mcq` — synthetic educational QA / multiple-choice signal for structured knowledge patterns.
- `factual_restraint` — synthetic factual-restraint signal for cautious-answer behavior.

## Intended Use

Use these datasets as targeted signals in a broader pretraining or continued-pretraining data mix. They are not intended to be described as SFT, DPO, or standalone training datasets.
