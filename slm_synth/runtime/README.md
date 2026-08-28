# `slm_synth/runtime`

## Purpose

Shared mechanical runtime used by all five dataset pipelines.

This package deliberately does **not** define dataset prompts, schemas, quality criteria, or public row semantics.

## Contents

~~~text
runtime/
├── backend.py       # OpenRouter backend construction
├── batching.py      # chunking, batch splitting, exact-cardinality fill
├── stages.py        # concurrent execution, retry, recursive isolation
├── novelty.py       # exact and near-duplicate detection
├── io.py            # JSON/JSONL helpers
└── reporting.py     # shared evidence/token reporting helpers
~~~

## Key Files

| File | Responsibility |
|---|---|
| `backend.py` | Builds the repo OpenRouter backend with configurable model, provider routing, sampling, and adaptive in-flight limits. |
| `batching.py` | Validates batch sizes, splits failed batches, and fills only missing cardinality. |
| `stages.py` | Executes model stages concurrently; after bounded retries it recursively isolates failing items and records persistent failures. |
| `novelty.py` | Normalized exact match, shingle/Jaccard lookup, and bounded sequence-similarity checks. |
| `io.py` | Small run-owned JSON/JSONL persistence primitives. |
| `reporting.py` | Extracts deterministic/quality evidence from manifests and provides SFT/DPO token estimates for reports. |

## How It Fits In

Every dataset `pipeline.py` imports runtime primitives but supplies its own prompts and semantic callbacks.

See [Architecture](../../docs/ARCHITECTURE.md).

## Usage

Example backend construction:

~~~python
from slm_synth.runtime import build_backend

backend = build_backend(
    model="deepseek/deepseek-v4-flash",
    max_tokens=4096,
    concurrency=8,
    routing_mode="auto",
)
~~~

Example novelty filtering:

~~~python
from slm_synth.runtime import NoveltyFilter

novelty = NoveltyFilter(jaccard_threshold=0.82, sequence_threshold=0.90)
ok, reason, nearest = novelty.check("candidate task")
if ok:
    novelty.add("task-001", "candidate task")
~~~

## Conventions

- Runtime code must be semantically neutral across datasets.
- Bounded retry/isolation is preferred over aborting an entire run because one batch failed.
- Cardinality-fill logic requests only the missing amount; it must not silently fabricate rows.
- Provider routing remains configurable; do not hardcode a provider in runtime logic.
- Reasoning suitability is enforced by the backend/model-suitability layer, not by dataset prompts.

## Gotchas

`runtime` is not a compatibility layer. If a helper is only useful to one dataset, keep it in that dataset package rather than expanding this package.
