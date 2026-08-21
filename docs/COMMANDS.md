# Command Reference

Lookup reference for supported Make targets and common variables. For end-to-end run order across all generation surfaces, see `GENERATION_WORKFLOW.md`.

## Command Groups

| Workflow | Smoke Run | Target Run | Inspect | Push |
|---|---|---|---|---|
| Pretraining | `make pretrain-smoke` | `make pretrain-generate` | `make pretrain-inspect` | `make pretrain-push` |
| SFT | `make sft-smoke` | `make sft-generate` | `make sft-inspect` | `make sft-push` |
| DPO | `make dpo-smoke` | `make dpo-generate` | `make dpo-inspect` | `make dpo-push` |
| Distillation SFT | `make distillation-sft-smoke` | `make distillation-sft-generate` | `make distillation-sft-inspect` | `make distillation-sft-push` |
| Distillation DPO | `make distillation-dpo-smoke` | `make distillation-dpo-generate` | `make distillation-dpo-inspect` | `make distillation-dpo-push` |

Run `make help` to print the command surface from the Makefile.

## Shared Live-Generation Variables

| Variable | Default | Purpose |
|---|---:|---|
| `PYTHON` | `python` | Python executable. |
| `MODEL` | `deepseek/deepseek-v4-flash` | Validated default OpenRouter model for live generation. |
| `MAX_TOKENS` | `4096` | Shared token default for commands that use it. |
| `OPENROUTER_ROUTING_MODE` | `auto` | Routing policy: `auto`, `prefer`, or `strict`. |
| `OPENROUTER_PROVIDER` | unset | Provider slug used by `prefer` or `strict` routing. |
| `HF_NAMESPACE` | `tohio` | Default Hugging Face namespace for dataset-specific push targets. |
| `HF_REPO` | unset | Explicit Hugging Face destination for push targets that use one repo id. |
| `HF_PRIVATE` | unset | Set to `true`, `yes`, or `1` for private Hugging Face repos. |

OpenRouter routing defaults to `auto`. Use `prefer` to try one provider first while allowing fallback, or `strict` to require one provider. `prefer` and `strict` require `OPENROUTER_PROVIDER`.

`MODEL` is a runtime default, not a fixed allowlist. Override it globally for a command with `MODEL=<openrouter-model-id>`, or override one workflow with variables such as `SFT_MODEL` or `DPO_MODEL`. Models outside the validated registry are allowed but emit a warning.

For providers that throttle, keep `OPENROUTER_ROUTING_MODE=auto` or use `prefer`; both retain fallback. `strict` disables fallback and can turn provider throttling into a terminal run failure.

Qualify model compatibility and role behavior before a smoke run:

```bash
QUALIFY_MODEL=openai/gpt-oss-20b QUALIFY_ROLES=sft-generator,sft-judge,sft-reviewer make model-qualify
make model-qualify-pretrain QUALIFY_MODEL=<model>
make model-qualify-alignment QUALIFY_MODEL=<model>
make model-qualify-distillation QUALIFY_MODEL=<model>
make model-qualify-all QUALIFY_MODEL=<model>
```

Qualification uses the same minimal portable contract as production: ordinary
messages plus max output and routing preferences. Provider-side JSON Schema,
tool choice, reasoning controls, temperature, and top-p are not required.

Estimate low, expected, and high generator/judge/reviewer costs from live
OpenRouter pricing before setting candidate budgets:

```bash
COST_GENERATOR_MODEL=<model> COST_JUDGE_MODEL=<model> \
COST_REVIEWER_MODEL=<model> COST_CANDIDATES=1000 make estimate-generation-cost

COST_GENERATOR_MODEL=<model> COST_JUDGE_MODEL=<model> \
COST_REVIEWER_MODEL=<model> COST_TARGET_TOKENS=100000 \
COST_AVERAGE_ACCEPTED_TOKENS=500 make estimate-generation-cost
```

The token-target form converts the desired accepted output into an estimated
accepted-record count, then accounts for judge rejection, reviewer disagreement,
malformed-output retries, and each role's input and output token profile.

## Pretraining

```bash
make pretrain-smoke
make pretrain-inspect
```

```bash
PRETRAIN_TARGET_TOKENS=1000000 PRETRAIN_TARGET_CONCURRENCY=4 make pretrain-generate
make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-target-001
```

`PRETRAIN_TOKENS` and `PRETRAIN_TARGET_TOKENS` are accepted public-text
targets. Generation automatically replaces validation and deduplication losses
with unused candidates. A run that exhausts its unique candidate inventory
before reaching the target fails with an accepted-token shortfall report; it is
not reported as complete.

| Variable | Default | Purpose |
|---|---:|---|
| `PRETRAIN_RUN` | `pretrain-smoke-001` | Smoke run id. |
| `PRETRAIN_TARGET_RUN` | `pretrain-target-001` | Target run id. |
| `PRETRAIN_TOKENS` | `100000` | Smoke token target. |
| `PRETRAIN_TARGET_TOKENS` | `1000000` | Target token target. |
| `PRETRAIN_BATCH_SIZE` | `32` | Maximum rows per provider request. |
| `PRETRAIN_CONCURRENCY` | `1` | Smoke request concurrency. |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | Target request concurrency. |
| `PRETRAIN_MODEL` | `$(MODEL)` | Pretraining model. |
| `PRETRAIN_SIGNAL` | unset | Optional single-signal filter. |
| `PRETRAIN_DIVERSITY_SAMPLE_SIZE` | `10000` | Deterministic diversity-audit sample limit per signal. |
| `PRETRAIN_DIVERSITY_THRESHOLD` | `0.80` | Jaccard threshold for five-token near-duplicate clusters. |

`make pretrain-report` also writes
`manifests/diversity_report_<stage>.json`. The diversity report measures
normalized template reuse, near-duplicate clusters, artifact-family
concentration, and cross-signal exact-template overlap. It is diagnostic and
does not mutate rows. The deduped-stage report fails if repetition remains,
and `pretrain-push` repeats a full-file exact/near-duplicate audit before any
remote commit.
It fails before reporting if `accepted_token_report.json` is missing, incomplete,
or contains a nonzero deficit for any configured signal.

## SFT

Audit every generic SFT and DPO source before spending on teacher generation:

```bash
make alignment-preflight
# or: make sft-preflight / make dpo-preflight
```

The same full-inventory gate runs inside the SFT and DPO generation entry
points before an OpenRouter backend is constructed.

```bash
make sft-smoke
make sft-inspect
```

```bash
SFT_FAMILIES="grounded_qa_and_reading rewriting_and_editing" \
SFT_CANDIDATE_COUNTS="grounded_qa_and_reading=2 rewriting_and_editing=2" \
SFT_GENERATION_RUN=sft-candidate-001 \
make sft-generate
make sft-inspect SFT_INSPECT_RUN=sft-candidate-001
```

Every selected family must appear exactly once in `SFT_CANDIDATE_COUNTS`.
Accepted rows are the quality-filtered result and rejected candidates are not
replaced to fill a quota.

| Variable | Default | Purpose |
|---|---:|---|
| `SFT_RUN` | `sft-smoke-001` | Smoke run id. |
| `SFT_GENERATION_RUN` | `sft-candidate-001` | Candidate generation run id. |
| `SFT_SMOKE_FAMILIES` | `grounded_qa_and_reading` | Smoke task-family list. |
| `SFT_FAMILIES` | `all` | Target family list. |
| `SFT_SMOKE_CANDIDATE_COUNTS` | `grounded_qa_and_reading=2` | Explicit smoke candidate plan. |
| `SFT_CANDIDATE_COUNTS` | unset | Required `family=count` candidate plan for `sft-generate`. |
| `SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request. |
| `SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `SFT_GENERATION_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests. |
| `SFT_RUN_ROOT` | `data/sft/runs` | Run output root. |
| `SFT_MODEL` | `$(MODEL)` | Teacher model. |
| `SFT_ADJUDICATOR_MODEL` | `$(SFT_MODEL)` | Independent semantic adjudicator model. |
| `SFT_REVIEWER_MODEL` | `$(SFT_ADJUDICATOR_MODEL)` | Independent reviewer of judge-accepted rows. |
| `SFT_REVIEWER_MAX_TOKENS` | `512` | Maximum reviewer output; reviewer output is intentionally small. |
| `SFT_ADJUDICATOR_MAX_TOKENS` | `$(SFT_MAX_TOKENS)` | Maximum adjudication completion tokens. |
| `SFT_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Holdout registry required by generation and reporting. |
| `SFT_HF_REPO` | `<HF_NAMESPACE>/slm-synthetic-sft` | One consolidated generic SFT dataset repository. |

`make sft-report` is a publication gate, not only a coverage summary. It
blocks exact and `0.88` Jaccard near-duplicate prompts/conversations, repeated
assistant responses, templates exceeding 40% of accepted rows, invalid public
role/tool sequences, missing or failed semantic-adjudication evidence, and
holdout collisions. `sft-push` rechecks every file-derived diagnostic against
the current JSONL files before creating a remote commit.

## DPO

```bash
make dpo-smoke
make dpo-inspect
```

```bash
DPO_PREFERENCE_DIMENSIONS="helpfulness_and_completeness instruction_adherence" \
DPO_CANDIDATE_COUNTS="helpfulness_and_completeness=40 instruction_adherence=40" \
DPO_GENERATION_RUN=dpo-candidate-001 make dpo-generate
make dpo-inspect DPO_INSPECT_RUN=dpo-candidate-001
```

Candidate counts bound paid work. Rejected and duplicate candidates are not
replaced. Accepted pairs and estimated tokens are the run outcome.

| Variable | Default | Purpose |
|---|---:|---|
| `DPO_RUN` | `dpo-smoke-001` | Smoke run id. |
| `DPO_GENERATION_RUN` | `dpo-candidate-001` | Candidate generation run id. |
| `DPO_SMOKE_PREFERENCE_DIMENSIONS` | `instruction_adherence` | Smoke preference-dimension list. |
| `DPO_PREFERENCE_DIMENSIONS` | `all` | Target preference-dimension list. |
| `DPO_SMOKE_CANDIDATE_COUNTS` | `instruction_adherence=2` | Explicit smoke candidate plan. |
| `DPO_CANDIDATE_COUNTS` | unset | Required `dimension=count` candidate plan. |
| `DPO_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request. |
| `DPO_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `DPO_GENERATION_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Candidate-run parallel requests. |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Run output root. |
| `DPO_MODEL` | `$(MODEL)` | Teacher model. |
| `DPO_ADJUDICATOR_MODEL` | `$(DPO_MODEL)` | Independent preference adjudicator model. |
| `DPO_REVIEWER_MODEL` | `$(DPO_ADJUDICATOR_MODEL)` | Independent reviewer of judge-accepted pairs. |
| `DPO_REVIEWER_MAX_TOKENS` | `512` | Maximum reviewer output. |
| `DPO_ADJUDICATOR_MAX_TOKENS` | `$(DPO_MAX_TOKENS)` | Maximum adjudication completion tokens. |
| `DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Holdout registry required by generation and reporting. |
| `DPO_HF_REPO` | `<HF_NAMESPACE>/slm-synthetic-dpo` | One consolidated generic DPO dataset repository. |

## Distillation SFT

```bash
make distillation-sft-smoke
make distillation-sft-inspect
```

```bash
DISTILLATION_SFT_SIGNALS="cloud code debugging" \
DISTILLATION_SFT_CANDIDATE_COUNTS="cloud=2 code=2 debugging=2" \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-candidate-001 \
make distillation-sft-generate
make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-candidate-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-candidate-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_SFT_RUN` | `distillation-sft-smoke-001` | Smoke run id. |
| `DISTILLATION_SFT_GENERATION_RUN` | `distillation-sft-candidate-001` | Candidate generation run id. |
| `DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL` | `2` | Smoke candidates per signal. |
| `DISTILLATION_SFT_CANDIDATE_COUNTS` | unset | Required `signal=count` candidate plan for production generation. |
| `DISTILLATION_SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum prompts per teacher request. |
| `DISTILLATION_SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests. |
| `DISTILLATION_SFT_GENERATION_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests. |
| `DISTILLATION_SFT_RUN_ROOT` | `data/distillation/runs` | Run output root. |
| `DISTILLATION_SFT_SIGNALS` | unset | Optional signal list. |
| `DISTILLATION_SFT_MODEL` | `$(MODEL)` | Teacher model. |
| `DISTILLATION_SFT_ADJUDICATIONS` | unset | Required path to reviewed member-level decisions. |
| `DISTILLATION_SFT_ADJUDICATION_RUN` | report run | Run receiving adjudication decisions. |

Repeated-response review uses the `member_fingerprint` values written to
`coverage.json`. Every member of every unresolved cluster needs an explicit
decision and reason:

```json
{
  "schema_version": 1,
  "decisions": [
    {
      "member_fingerprint": "<64-character fingerprint>",
      "decision": "keep",
      "reason": "response matches this prompt"
    }
  ]
}
```

Apply reviewed decisions locally, then rebuild the report:

```bash
make distillation-sft-adjudicate \
  DISTILLATION_SFT_ADJUDICATION_RUN=distillation-sft-candidate-001 \
  DISTILLATION_SFT_ADJUDICATIONS=adjudications/distillation-sft-candidate-001.json

make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-candidate-001
make distillation-sft-push DISTILLATION_SFT_PUSH_RUN=distillation-sft-candidate-001
```

Adjudication preserves rejected rows under the run's `rejected/` directory.
It updates the run manifest with generated, curated, and rejected counts.
Publication remains blocked for unresolved clusters or inconsistent manifest
counts; rejected rows do not create a replacement quota.

## Distillation DPO

```bash
make distillation-dpo-smoke
make distillation-dpo-inspect
```

```bash
DISTILLATION_DPO_TARGET_PAIRS=15000 DISTILLATION_DPO_TARGET_RUN=distillation-dpo-target-001 make distillation-dpo-generate
make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-target-001
```

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_DPO_RUN` | `distillation-dpo-smoke-001` | Smoke run id. |
| `DISTILLATION_DPO_TARGET_RUN` | `distillation-dpo-target-001` | Target run id. |
| `DISTILLATION_DPO_SMOKE_FAMILIES` | `teacher_response_preference` | Smoke family list. |
| `DISTILLATION_DPO_FAMILIES` | `all` | Target family list. |
| `DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY` | `1000` | Smoke accepted pairs for the single Distillation-DPO family. |
| `DISTILLATION_DPO_TARGET_PAIRS` | `15000` | Production accepted preference-pair target. |
| `DISTILLATION_DPO_RUN_ROOT` | `data/distillation-dpo/runs` | Run output root. |
| `DISTILLATION_DPO_MODEL` | `$(MODEL)` | Teacher lineage model recorded in manifests. |
| `DISTILLATION_DPO_MAX_BACKFILL_ROUNDS` | `2` | Accepted-target backfill budget after pair quality gates. |
| `DISTILLATION_DPO_HOLDOUT_REGISTRY` | `configs/eval_holdouts.yaml` | Holdout registry required by generation and reporting. |

## Reports and Publishing

```bash
make pretrain-report PRETRAIN_REPORT_RUN=<run-id>
make sft-report SFT_REPORT_RUN=<run-id>
make dpo-report DPO_REPORT_RUN=<run-id>
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=<run-id>
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=<run-id>
```

```bash
make pretrain-push HF_REPO=<namespace>/<repo>
make sft-push SFT_HF_REPO=<namespace>/<repo>
make dpo-push DPO_HF_REPO=<namespace>/<repo>
make distillation-sft-push DISTILLATION_SFT_HF_REPO=<namespace>/<repo>
make distillation-dpo-push DISTILLATION_DPO_HF_NAMESPACE=<namespace>
```

`sft-push` creates one atomic version in `SFT_HF_REPO`. The default dataset
configuration loads every `data/<task_family>.jsonl` file; named task-family
configurations load one file. Only flat final files are accepted.

`dpo-push` creates one atomic version in `DPO_HF_REPO`. The default
configuration loads every `data/<preference_dimension>.jsonl` file; each named
configuration loads one dimension file. Only flat final files are accepted.

## Hugging Face Dataset Deletion

Deletion targets remove Hugging Face **dataset repositories** only. They do not delete local run data under `data/`.

All deletion targets are dry-run by default. Set `HF_DELETE_YES=1` to actually delete selected repos.

### Dry-run examples

```bash
make hf-delete-distillation
make hf-delete-legacy-distillation-dpo
```

Delete target output should list selected repositories and print `DRY RUN ONLY`.

### Actual deletion examples

```bash
HF_DELETE_YES=1 make hf-delete-distillation
HF_DELETE_YES=1 make hf-delete-legacy-distillation-dpo
```

Delete an exact repo:

```bash
make hf-delete-datasets HF_DELETE_REPO=tohio/slm-synthetic-distillation-dpo
HF_DELETE_YES=1 make hf-delete-datasets HF_DELETE_REPO=tohio/slm-synthetic-distillation-dpo
```

Delete repos from a file:

```bash
make hf-delete-datasets HF_DELETE_REPO_FILE=repos-to-delete.txt
HF_DELETE_YES=1 make hf-delete-datasets HF_DELETE_REPO_FILE=repos-to-delete.txt
```

The repo file should contain one dataset repo id per line:

```text
tohio/slm-synthetic-distillation-sft
tohio/slm-synthetic-distillation-dpo
```

### Delete targets

| Target | Purpose |
|---|---|
| `make hf-delete-datasets` | Delete exact repos from `HF_DELETE_REPO` or `HF_DELETE_REPO_FILE`. |
| `make hf-delete-distillation` | Delete `slm-synthetic-distillation-sft` and `slm-synthetic-distillation-dpo`. |
| `make hf-delete-legacy-distillation-dpo` | Delete the old long distillation-DPO repo name. |

### Delete variables

| Variable | Default | Purpose |
|---|---:|---|
| `HF_DELETE_NAMESPACE` | `$(HF_NAMESPACE)` | Hugging Face namespace for generated delete targets. |
| `HF_DELETE_REPO` | unset | Exact dataset repo id to delete. |
| `HF_DELETE_REPO_FILE` | unset | File containing one exact dataset repo id per line. |
| `HF_DELETE_YES` | unset | Set to `1`, `true`, or `yes` to actually delete. |

Actual deletion requires `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` in the environment.

## Maintenance

```bash
make test
make clean
```
