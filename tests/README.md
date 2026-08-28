# `tests`

## Purpose

Validate the repository's production contracts: five dataset pipelines, shared runtime mechanics, model suitability, public schemas, reports/manifests, holdout behavior, and publication boundaries.

Tests are primarily deterministic/local. They do not replace live provider smoke runs.

## Contents

Test files are grouped by the subsystem or dataset they exercise:

~~~text
test_pretrain_* / test_grounded_*      pretraining and grounded artifacts
test_sft_*                             generic SFT
test_dpo_*                             generic DPO
test_distillation_*                    both distillation products
test_model_*                           model suitability / qualification contracts
test_openrouter_*                      routing and backend behavior
test_*manifest* / test_*report*        evidence, reporting, publication readiness
~~~

## How It Fits In

`make test` first compiles `slm_synth` and `tests`, then runs pytest.

~~~bash
make test
~~~

Focused examples:

~~~bash
pytest -q tests/test_pretrain_*.py tests/test_grounded_*.py
pytest -q tests/test_sft_*.py
pytest -q tests/test_dpo_*.py
pytest -q tests/test_distillation_*.py
pytest -q tests/test_model_*.py
~~~

## What Tests Should Protect

Tests should assert current production behavior, including:

- exact public row schemas;
- deterministic validation evidence;
- judge/reviewer evidence completeness;
- final acceptance/dedup rules;
- Make target → pipeline wiring;
- holdout collision rejection;
- consolidated publication layout;
- reasoning suitability policy;
- runtime retry/splitting/cardinality behavior.

Tests should not preserve deleted compatibility commands or legacy orchestration APIs.

## Live Validation

A passing test suite does not prove a provider/model works live. For provider/model changes:

~~~bash
QUALIFY_MODEL=<model-id> make model-qualify-sft
make sft-smoke
~~~

Use the dataset-specific smoke target for the path you changed.

## Public Row Boundaries

| Dataset | Public training fields |
|---|---|
| SFT | `id`, `messages`, `metadata` (+ schema-supported optional interaction fields) |
| DPO | `id`, `prompt`, `chosen`, `rejected`, `metadata` |
| Distillation SFT | `id`, `prompt`, `reasoning`, `response`, `metadata` |
| Distillation DPO | `id`, `prompt`, `chosen`, `rejected`, `metadata` |

Provider, routing, retry, cost, batch, and private lineage fields belong in manifests/reports rather than public training rows.

## Gotchas

If pytest fails during collection because `openai`, `datasketch`, or another pinned dependency is missing, install `requirements.txt` before diagnosing repository code.
