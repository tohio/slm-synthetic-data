# Troubleshooting

How to diagnose generation, provider, quality, reporting, and publication failures without bypassing the production pipeline.

## Start With the Failing Stage

Do not delete a run immediately. The work/manifests directories are intended to explain why a run failed.

For SFT/DPO/distillation runs, inspect:

~~~text
<run>/work/<family-or-dimension>/
<run>/internal/<signal>/            # Distillation SFT
<run>/manifests/
<run>/coverage.json
~~~

Common evidence files include:

~~~text
stage.failures.jsonl
quality.rejected.jsonl
judge.decisions.jsonl
reviewer.decisions.jsonl
summary.json
~~~

For pretraining, inspect:

~~~text
<run>/manifests/accepted_token_report.json
<run>/manifests/quality/
<run>/quality_accepted/
<run>/rejected/
~~~

## Provider 429 / Throttling / Endpoint Failures

Symptoms:

- repeated provider retries;
- batches split to smaller sizes;
- one provider consistently fails while another succeeds;
- OpenRouter reports no compatible route.

Use `auto` routing unless you have a reason to pin:

~~~bash
OPENROUTER_ROUTING_MODE=auto make sft-smoke
~~~

Prefer one provider while retaining fallback:

~~~bash
OPENROUTER_ROUTING_MODE=prefer \
OPENROUTER_PROVIDER=DeepInfra \
make sft-smoke
~~~

If provider-specific throttling is common, preserve a provider order:

~~~bash
OPENROUTER_PROVIDER_ORDER="Baidu,CoreWeave,DeepInfra" make sft-smoke
~~~

Do not remove retry/isolation logic to make a flaky provider appear successful.

## Model Returns Invalid Structured Output

Run model qualification first:

~~~bash
QUALIFY_MODEL=<model-id> make model-qualify-sft
~~~

A model must support the production strict JSON Schema request path. A model that can answer normal chat prompts but cannot satisfy the structured contract is not suitable for the role.

If failures are isolated to large batches, the runtime will retry and split them automatically. Persistent single-item failures are recorded in `stage.failures.jsonl`.

## Mandatory Reasoning Model Rejected

This is expected behavior.

Synthetic generation roles require visible training content without mandatory hidden/extra reasoning behavior. The suitability layer:

1. detects reasoning capability;
2. disables reasoning when supported;
3. verifies the structured request succeeds with reasoning disabled;
4. rejects models where reasoning cannot be disabled.

Do not bypass this check by calling the provider directly from a dataset pipeline.

## Too Many Near-Duplicate Tasks

Check the relevant work summary and novelty thresholds.

Default task novelty thresholds for SFT/DPO/distillation are:

~~~text
Jaccard: 0.82
Sequence similarity: 0.90
~~~

A high duplicate rate usually means the derivation/task prompt is collapsing onto a small template family. Lowering the threshold only makes filtering stricter; raising it allows more similarity. Change thresholds only after inspecting examples.

## Judge Acceptance Is Very Low

Inspect `quality.rejected.jsonl` and `judge.decisions.jsonl`.

Distinguish:

- generator defects;
- deterministic validation defects;
- genuinely poor content;
- judge calibration problems;
- schema/format failures.

Do not tune for 100% acceptance. Rejection is expected when quality gates are functioning.

## Reviewer Rejects Judge-Accepted Rows

This is also expected. Reviewer is intentionally independent.

For Distillation SFT, pay particular attention to:

- unsupported reviewer claims;
- contradiction between defect category and verification;
- code-review requirements invented by the reviewer;
- repeated/generic teacher responses.

For Distillation DPO, confirm the five judge gates were satisfied before reviewing reviewer behavior.

## Pretraining Is Under Token Target

Read:

~~~text
data/runs/<run>/manifests/accepted_token_report.json
~~~

The relevant number is **post-review accepted tokens**, not raw generated tokens.

The pipeline automatically estimates survival yield and backfills through the same generation → deterministic validation → judge → reviewer → final dedup path up to `PRETRAIN_MAX_BACKFILL_ROUNDS`.

If it exhausts backfill rounds:

- inspect rejection rates by signal;
- inspect duplicate rates;
- inspect accepted average length;
- verify the target is realistic for the configured grounded artifact capacity;
- increase backfill rounds only after understanding the deficit.

## Holdout Collision

Generic SFT, generic DPO, and Distillation DPO can reject rows that collide with `configs/eval_holdouts.yaml`.

Do not disable holdout checks to publish a run. Change the generated task/source so it no longer matches an exact evaluation prompt or registered holdout key.

## Report Says Evidence Is Missing

A supported production run writes deterministic and semantic evidence as it generates.

If a report says evidence is missing, common causes are:

- the run predates the current pipeline;
- a run directory was partially copied;
- a manifest was manually edited;
- generation was interrupted before final manifest writing;
- a legacy command produced the data.

Regenerate through the supported `<dataset>-smoke` or `<dataset>-generate` target rather than fabricating evidence.

## Push Is Blocked

Push commands intentionally fail closed when the run is not publication-ready.

Check:

- final dataset files exist;
- run manifest exists;
- coverage report is current;
- accepted counts match public files;
- deterministic/semantic evidence is complete;
- holdout checks pass;
- HF repo id/token are correct.

Do not upload an internal `work/`, `internal/`, raw, rejected, or batch-shard file as public training data.

## `make test` Fails During Collection

Install the repository-pinned requirements:

~~~bash
pip install -r requirements.txt
~~~

`openai` and `datasketch` are required by runtime/provider and diversity tests. A missing dependency during collection is an environment problem, not evidence that a dataset pipeline is broken.

## See Also

- [Command Reference](COMMANDS.md)
- [Generation Workflow](GENERATION_WORKFLOW.md)
