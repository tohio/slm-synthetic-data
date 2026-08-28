# Generation Workflow

Operational runbook from model qualification through smoke validation, production generation, reporting, inspection, and publication.

## Before a Live Run

### 1. Install dependencies and configure credentials

~~~bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
~~~

At minimum, set `OPENROUTER_API_KEY`. Set `HF_TOKEN` before publication.

### 2. Qualify changed models

The repository defaults are known production roles, but any replacement model should be qualified before a generation run.

~~~bash
QUALIFY_MODEL=<model-id> make model-qualify-sft
~~~

Qualification validates:

- the OpenRouter request path,
- strict structured output,
- required JSON Schema behavior,
- role-specific output shape,
- reasoning disabled where supported,
- rejection of mandatory-reasoning models.

### 3. Run smoke before production

A smoke run is a live integration test of provider routing, model contracts, local validation, judge/reviewer stages, output writing, reports, and manifests.

Do not treat `make test` as a substitute for a smoke run.

---

## Common Stage Mechanics

The SFT/DPO/distillation pipelines share the same runtime behavior:

1. work is batched;
2. model calls run concurrently up to the configured limit;
3. a failed batch is retried a bounded number of times;
4. persistent multi-item failures are recursively split;
5. persistent single-item failures are recorded rather than aborting unrelated work;
6. cardinality-fill attempts request only missing items;
7. novelty filters reject exact/near duplicates before expensive downstream stages;
8. deterministic validation runs before judge;
9. reviewer sees judge-accepted candidates only;
10. final dedup produces the public accepted set.

Runtime failures are recorded in per-run work artifacts so a low-quality or malformed batch is diagnosable.

---

## Pretraining Runbook

### Flow

~~~text
grounded artifact
→ model rendering
→ deterministic validation
→ semantic judge
→ reviewer
→ final global dedup
→ accepted-token calculation
→ backfill if under target
→ final pretrain.jsonl
~~~

### Smoke

~~~bash
PRETRAIN_RUN=pretrain-smoke-001 make pretrain-smoke
~~~

The smoke command:

1. writes `configs/synthetic.yaml` from `configs/synthetic_template.yaml`;
2. preflights deterministic grounded artifacts;
3. runs the integrated pretraining pipeline;
4. writes grounded artifact reporting;
5. runs final report/card/manifest normalization.

Primary final artifact:

~~~text
data/runs/<run>/deduped/pretrain.jsonl
~~~

Important supporting artifacts include:

~~~text
data/runs/<run>/manifests/accepted_token_report.json
data/runs/<run>/manifests/quality/
data/runs/<run>/quality_accepted/
data/runs/<run>/rejected/
data/runs/<run>/README.md
~~~

### Completion Rule

Pretraining is complete only when **post-review, post-dedup accepted tokens** meet the target. A row rejected by deterministic validation, judge, reviewer, or final dedup contributes zero accepted tokens.

If the final accepted set is under target, the pipeline increases the generation target and routes new candidates through the same complete quality path. Backfill cannot bypass semantic review.

### Inspect

~~~bash
PRETRAIN_INSPECT_RUN=<run> make pretrain-inspect
~~~

Check:

- final rows in `deduped/pretrain.jsonl`;
- accepted-token target and deficit;
- quality acceptance counts by signal;
- rejection reasons;
- provider/adaptive batching telemetry in generation manifests;
- diversity report.

### Production

~~~bash
PRETRAIN_TARGET_RUN=pretrain-production-001 \
PRETRAIN_TARGET_TOKENS=<accepted-token-target> \
make pretrain-generate
~~~

### Publish

~~~bash
HF_REPO=<namespace>/<repo> make pretrain-push
~~~

`pretrain-push` reads the output directory from the current `configs/synthetic.yaml`; it does not use `PRETRAIN_REPORT_RUN` as a run selector. Publication verifies semantic-quality completion before upload.

---

## Generic SFT Runbook

### Flow

~~~text
derivation
→ concrete task
→ task novelty
→ answer
→ deterministic validation
→ judge
→ reviewer
→ final exact dedup
~~~

### Smoke

~~~bash
SFT_RUN=sft-smoke-001 make sft-smoke
~~~

By default smoke uses `grounded_qa_and_reading` with a very small derivation/task plan.

Override the family to exercise a different capability:

~~~bash
SFT_FAMILIES=programming SFT_RUN=sft-programming-smoke-001 make sft-smoke
~~~

### Run Layout

~~~text
data/sft/runs/<run>/
├── datasets/
│   └── <family>.jsonl
├── manifests/
│   ├── <family>.<run>.manifest.json
│   └── <run>.manifest.json
├── work/
│   └── <family>/
│       ├── derivations.generated.jsonl
│       ├── tasks.generated.jsonl
│       ├── tasks.accepted.jsonl
│       ├── answers.generated.jsonl
│       ├── judge.decisions.jsonl
│       ├── reviewer.decisions.jsonl
│       ├── quality.rejected.jsonl
│       ├── stage.failures.jsonl
│       ├── sft.accepted.jsonl
│       └── summary.json
├── coverage.json
└── README.md
~~~

`work/` is internal evidence/debug state. `datasets/` contains the public training rows.

### Inspect and Report

~~~bash
SFT_INSPECT_RUN=<run> make sft-inspect
SFT_REPORT_RUN=<run> make sft-report
~~~

Look for:

- family coverage and accepted counts;
- deterministic-validation evidence;
- judge/reviewer evidence;
- duplicate rejection counts;
- holdout collisions;
- missing/underfilled families.

### Production and Publish

~~~bash
SFT_GENERATION_RUN=sft-production-001 make sft-generate

SFT_REPORT_RUN=sft-production-001 make sft-report
SFT_PUSH_RUN=sft-production-001 make sft-push
~~~

One Hugging Face repository contains all selected family JSONL files. Family configurations are views over the same repository, not separate repositories.

---

## Generic DPO Runbook

### Flow

~~~text
derivation
→ concrete task
→ task novelty
→ chosen/rejected pair
→ deterministic pair validation
→ judge
→ reviewer
→ final exact triple dedup
~~~

### Smoke

~~~bash
DPO_RUN=dpo-smoke-001 make dpo-smoke
~~~

Default dimension: `instruction_adherence`.

Override it:

~~~bash
DPO_PREFERENCE_DIMENSIONS=code_correctness \
DPO_RUN=dpo-code-smoke-001 \
make dpo-smoke
~~~

### Run Layout

~~~text
data/dpo/runs/<run>/
├── datasets/
│   └── <dimension>.jsonl
├── manifests/
├── work/
│   └── <dimension>/
│       ├── derivations.generated.jsonl
│       ├── tasks.generated.jsonl
│       ├── tasks.accepted.jsonl
│       ├── pairs.generated.jsonl
│       ├── pairs.validated.jsonl
│       ├── judge.decisions.jsonl
│       ├── reviewer.decisions.jsonl
│       ├── quality.rejected.jsonl
│       ├── stage.failures.jsonl
│       ├── dpo.accepted.jsonl
│       └── summary.json
├── coverage.json
└── README.md
~~~

Internal generation uses plain strings for prompt/chosen/rejected. Final public rows are converted to the message-list DPO schema before writing `datasets/`.

### Production and Publish

~~~bash
DPO_GENERATION_RUN=dpo-production-001 make dpo-generate
DPO_REPORT_RUN=dpo-production-001 make dpo-report
DPO_PUSH_RUN=dpo-production-001 make dpo-push
~~~

The report/publish path checks evaluation holdouts through `configs/eval_holdouts.yaml`.

---

## Distillation SFT Runbook

### Flow

~~~text
derivation
→ student-appropriate prompt
→ prompt novelty
→ teacher response
→ deterministic validation
→ response novelty
→ judge
→ reviewer
→ prompt/response dedup
~~~

### Why Response Novelty Is Separate

A response can be locally valid yet still be bad distillation data if the same generic answer is reused across unrelated prompts. Distillation SFT therefore applies response-level exact/near-duplicate checks before judge and exact prompt/response uniqueness at final acceptance.

### Smoke

~~~bash
DISTILLATION_SFT_RUN=distillation-sft-smoke-001 \
make distillation-sft-smoke
~~~

Default signal: `debugging`.

### Run Layout

~~~text
data/distillation/runs/<run>/
├── datasets/
│   └── <signal>.jsonl
├── manifests/
│   ├── <signal>.<run>.manifest.json
│   └── <run>.manifest.json
├── internal/
│   └── <signal>/
│       ├── derivations.generated.jsonl
│       ├── tasks.generated.jsonl
│       ├── tasks.accepted.jsonl
│       ├── answers.generated.jsonl
│       ├── judge.decisions.jsonl
│       ├── reviewer.decisions.jsonl
│       ├── quality.rejected.jsonl
│       ├── stage.failures.jsonl
│       ├── distillation_sft.accepted.jsonl
│       └── summary.json
├── coverage.json
└── README.md
~~~

Public rows contain `reasoning: null`; private generation reasoning or provider metadata is never copied into public training rows.

### Production and Publish

~~~bash
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-production-001 \
make distillation-sft-generate

DISTILLATION_SFT_REPORT_RUN=distillation-sft-production-001 \
make distillation-sft-report

DISTILLATION_SFT_PUSH_RUN=distillation-sft-production-001 \
make distillation-sft-push
~~~

---

## Distillation DPO Runbook

### Flow

~~~text
derivation
→ task
→ task novelty
→ pair
→ deterministic pair validation
→ five-gate judge
→ reviewer
→ final exact triple dedup
~~~

### Five-Gate Judge

A pair is judge-accepted only when every field is true:

~~~text
assessable
chosen_complete
chosen_correct
preference_valid
dimension_aligned
~~~

Reviewer then independently checks only those judge-accepted pairs.

### Smoke

~~~bash
DISTILLATION_DPO_RUN=distillation-dpo-smoke-001 \
make distillation-dpo-smoke
~~~

Default dimension: `factual_accuracy`.

### Run Layout

~~~text
data/distillation-dpo/runs/<run>/
├── datasets/
│   └── teacher_response_preference.jsonl
├── manifests/
│   ├── teacher_response_preference.<run>.manifest.json
│   └── <run>.manifest.json
├── work/
│   └── <dimension>/
├── coverage.json
└── README.md
~~~

All selected preference dimensions are internal components of the single public `teacher_response_preference.jsonl` dataset.

### Production and Publish

~~~bash
DISTILLATION_DPO_TARGET_RUN=distillation-dpo-production-001 \
make distillation-dpo-generate

DISTILLATION_DPO_REPORT_RUN=distillation-dpo-production-001 \
make distillation-dpo-report

DISTILLATION_DPO_PUSH_RUN=distillation-dpo-production-001 \
make distillation-dpo-push
~~~

---

## What to Review Before Publication

For every dataset:

- public rows are structurally valid;
- expected families/signals/dimensions are represented;
- deterministic validation evidence exists;
- judge/reviewer evidence exists where required;
- final accepted counts match reports/manifests;
- holdout checks pass where configured;
- no internal provider/cost/retry fields leaked into public rows;
- dataset card and coverage report match the public files;
- the intended Hugging Face repo id is explicit.

## See Also

- [Command Reference](COMMANDS.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Architecture](ARCHITECTURE.md)
