# Generation Workflow

End-to-end run ladder for generating synthetic datasets safely.

## Run Ladder

Use the same order for every active generation surface:

1. Run a smoke job.
2. Inspect public rows or pairs.
3. Inspect run manifests, planning fields, telemetry, and public-directory hygiene.
4. Run a small target override.
5. Inspect the small-scale output.
6. Run the full target only after earlier outputs pass inspection.
7. Push only public artifacts.

Public dataset directories should contain final public files only. Batch shards, partial files, rejected rows, retry files, provider internals, and scratch files stay out of public upload discovery.

SFT and distillation SFT candidate plans limit generation work; accepted rows
are the quality-filtered outcome. Rejected and duplicate candidates are not
replaced to fill a quota. DPO pair planning remains separate.

## Generic SFT

Before a smoke or production request, run `make alignment-preflight`. The gate
loads all 60 SFT briefs and all 90 independently authored DPO prompts, even when
the next run requests only one family. Generation invokes the same check before
constructing its provider backend, so a bad catalog cannot consume paid calls.

Smoke:

```bash
make sft-smoke
make sft-inspect SFT_INSPECT_RUN=sft-smoke-001
```

Small candidate run:

```bash
SFT_FAMILIES="grounded_qa_and_reading rewriting_and_editing" \
SFT_CANDIDATE_COUNTS="grounded_qa_and_reading=2 rewriting_and_editing=2" \
SFT_GENERATION_RUN=sft-small-001 \
make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-small-001
make sft-report SFT_REPORT_RUN=sft-small-001
```

Another explicit candidate plan (choose counts only after reviewing family capacity and pilot quality):

```bash
SFT_FAMILIES="grounded_qa_and_reading rewriting_and_editing" \
SFT_CANDIDATE_COUNTS="grounded_qa_and_reading=3 rewriting_and_editing=3" \
SFT_GENERATION_RUN=sft-prod-001 \
make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-prod-001
make sft-report SFT_REPORT_RUN=sft-prod-001
```

Push after inspection:

```bash
make sft-push \
  SFT_PUSH_RUN=sft-prod-001 \
  SFT_HF_REPO=tohio/slm-synthetic-sft
```

Public rows are written under `data/sft/runs/<run>/datasets/`, with one final JSONL file per family. Batch shards remain under the sibling `batches/` directory.

### SFT acceptance

SFT accepts the first row for each normalized ID, prompt, and complete conversation. Duplicate rows and terminal local-validation failures are rejected. Repeated assistant responses are reported but are not automatically rejected because distinct factual prompts can share an answer. Every selected family requires an explicit candidate count; generation never infers an equal allocation from a global row target.

Conversation validation is strict. An optional system message may appear only
first; ordinary turns alternate user and assistant; tool calls use declared
shared tools and must be followed by matching tool responses and a final
assistant message. Adjacent roles and malformed tool cycles are not repaired.

### SFT reporting and publish blockers

`make sft-report` loads `configs/eval_holdouts.yaml` by default and writes
`coverage.json`. The report includes aggregate and per-family metadata
coverage, normalized ID/prompt/conversation uniqueness, exact and near
duplicate clusters, repeated assistant-response clusters, template
concentration, public-row validation, semantic-adjudication evidence, holdout
collisions, and candidate/attempted/accepted/rejected/duplicate counts.

Near duplicates use normalized token-set Jaccard similarity at `0.88`.
Publication also rejects any template used by more than 40% of the accepted
rows when it occurs more than once. These are fixed publication-quality rules,
not sizing knobs.

Publication is blocked when:

- IDs, normalized prompts, or normalized conversations repeat or form a
  near-duplicate pair.
- An assistant response repeats across accepted rows.
- A template exceeds the concentration limit.
- A public row has an invalid schema, role sequence, or tool-call lifecycle.
- Any published row lacks passing semantic-adjudication evidence in the batch
  manifests, or its recorded adjudication fails.
- Holdouts were not checked or a holdout collision exists.
- Candidate and acceptance accounting is inconsistent.
- `coverage.json` is missing or stale relative to public files.
- The dataset card lacks the default or any required family configuration.

### SFT publication migration

`sft-push` publishes one atomic commit to `SFT_HF_REPO`:

```text
tohio/slm-synthetic-sft/
├── README.md
├── data/
│   ├── grounded_qa_and_reading.jsonl
│   ├── rewriting_and_editing.jsonl
│   └── ...
└── artifacts/
    ├── coverage.json
    └── manifests/
```

The `default` configuration loads all `data/*.jsonl` files as the train split.
Each task-family configuration loads its one `data/<task_family>.jsonl` file;
it does not duplicate stored rows. The run workflow is the only generic SFT
generation path. Publication rejects batch shards, nested JSONL files, and
rows whose `task_family` does not match the filename.

## Generic DPO

Smoke:

```bash
make dpo-smoke
make dpo-inspect DPO_INSPECT_RUN=dpo-smoke-001
```

Small candidate run:

```bash
DPO_PREFERENCE_DIMENSIONS="instruction_adherence" DPO_CANDIDATE_COUNTS="instruction_adherence=100" \
DPO_GENERATION_RUN=dpo-small-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-small-001
make dpo-report DPO_REPORT_RUN=dpo-small-001
```

Larger candidate run:

```bash
DPO_PREFERENCE_DIMENSIONS="helpfulness_and_completeness factual_accuracy instruction_adherence" \
DPO_CANDIDATE_COUNTS="helpfulness_and_completeness=500 factual_accuracy=500 instruction_adherence=500" \
DPO_GENERATION_RUN=dpo-prod-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-prod-001
make dpo-report DPO_REPORT_RUN=dpo-prod-001
```

Push after inspection:

```bash
make dpo-push \
  DPO_PUSH_RUN=dpo-prod-001 \
  DPO_HF_REPO=tohio/slm-synthetic-dpo
```

Public pairs are written under `data/dpo/runs/<run>/datasets/`, with one final
JSONL file per preference dimension. Internal batch shards remain under the
sibling `batches/` directory and are never publication inputs.

### DPO candidate acceptance

DPO accepts the first pair for each normalized ID, prompt, and complete `(prompt, chosen, rejected)` triple. Duplicate pairs and terminal quality failures are recorded as outcomes of the explicit candidate inventory. Chosen/rejected similarity and repeated negative patterns remain diagnostic because controlled near-miss negatives can be intentional.

Each row has one shared prompt and optional tool inventory. Chosen and rejected
are explicit continuation branches and may contain multiple assistant/tool
messages. Both branches are validated independently against the same prompt and
tools; nested replacement prompts, undeclared calls, unresolved call IDs, and
malformed role sequences are rejected.

Generation attempts each declared candidate once. Adaptive splitting may isolate a failed multi-candidate provider response, but it does not add candidates or replace rejected pairs. Accepted pairs are the quality-filtered output.

### DPO reporting and publish blockers

`make dpo-report` loads `configs/eval_holdouts.yaml` and writes `coverage.json`. The report includes aggregate and per-dimension metadata coverage, normalized ID/prompt/triple uniqueness, chosen/rejected similarity, negative-pattern distribution, holdout collisions, candidate/attempted/accepted/rejected/duplicate counts, and estimated accepted tokens.

Publication is blocked when:

- IDs, normalized prompts, or normalized preference triples repeat.
- Holdouts were not checked or a collision exists.
- Candidate/accepted accounting is inconsistent with the public files.
- `coverage.json` is missing or stale relative to the public files.
- The dataset card lacks the default or any required preference-dimension configuration.

### DPO consolidated publication

`dpo-push` publishes one atomic commit to `DPO_HF_REPO`:

```text
tohio/slm-synthetic-dpo/
├── README.md
├── data/
│   ├── instruction_adherence.jsonl
│   ├── groundedness.jsonl
│   └── ...
└── artifacts/
    ├── coverage.json
    └── manifests/
```

The `default` configuration loads all `data/*.jsonl` files as the train split.
Each named preference-dimension configuration loads its one
`data/<preference_dimension>.jsonl` file without duplicating pairs. The run
workflow is the only generic DPO generation path. Publication rejects batch
shards, nested JSONL files, and rows whose `preference_dimension` does not
match the filename.

## Distillation SFT

Smoke:

```bash
make distillation-sft-smoke
make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-smoke-001
```

Small candidate run:

```bash
DISTILLATION_SFT_SIGNALS="cloud code" \
DISTILLATION_SFT_CANDIDATE_COUNTS="cloud=2 code=2" \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-small-001 \
make distillation-sft-generate

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-small-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-small-001
```

Another explicit candidate plan (choose counts only after reviewing signal capacity and pilot quality):

```bash
DISTILLATION_SFT_SIGNALS="cloud code debugging" \
DISTILLATION_SFT_CANDIDATE_COUNTS="cloud=4 code=4 debugging=4" \
DISTILLATION_SFT_GENERATION_RUN=distillation-sft-prod-001 \
make distillation-sft-generate

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-prod-001
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-prod-001
```

Review `coverage.json` before publishing. If it contains unresolved
`repeated_response_clusters`, record a `keep` or `reject` decision and reason
for every `member_fingerprint`, then apply the decisions:

```bash
make distillation-sft-adjudicate \
  DISTILLATION_SFT_ADJUDICATION_RUN=distillation-sft-prod-001 \
  DISTILLATION_SFT_ADJUDICATIONS=adjudications/distillation-sft-prod-001.json
```

Rejected rows remain under `rejected/`. They are not replaced to preserve a
nominal row target. Adjudication updates the run manifest's curated counts;
rebuild and inspect the report afterward.

```bash
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=distillation-sft-prod-001
make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-prod-001
```

Push only after the rebuilt report is accepted. Publication fails for
unresolved repeated-response clusters or stale manifest counts.

```bash
make distillation-sft-push DISTILLATION_SFT_PUSH_RUN=distillation-sft-prod-001
```

Public rows are written under `data/distillation/runs/<run>/datasets/` as per-signal JSONL files.

## Distillation DPO

Smoke:

```bash
make distillation-dpo-smoke
make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-smoke-001
```

The smoke target is 1,000 accepted preference pairs.

Generation preflights the complete initial-plus-backfill source range against
`configs/eval_holdouts.yaml`. Reporting records the holdout check, and publishing
is blocked when the check is missing, a collision exists, or the saved report is
stale for the current dataset files.

Production target:

```bash
DISTILLATION_DPO_TARGET_PAIRS=15000 DISTILLATION_DPO_TARGET_RUN=distillation-dpo-prod-001 make distillation-dpo-generate

make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-prod-001
make distillation-dpo-report DISTILLATION_DPO_REPORT_RUN=distillation-dpo-prod-001
```

Push after inspection:

```bash
make distillation-dpo-push DISTILLATION_DPO_PUSH_RUN=distillation-dpo-prod-001
```

Public pairs are written under `data/distillation-dpo/runs/<run>/datasets/`.
The production target is 15,000 accepted pairs. Duplicate prompts and preference
triples do not count toward that target. Response repetition, pair similarity,
and negative-construction patterns are reported for inspection but are not used
as automatic semantic judges.

## Pretraining

Smoke:

```bash
make pretrain-smoke
make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-smoke-001
```

Small target override:

```bash
PRETRAIN_TARGET_TOKENS=100000 PRETRAIN_TARGET_RUN=pretrain-small-001 make pretrain-generate

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-small-001
make pretrain-report PRETRAIN_REPORT_RUN=pretrain-small-001
```

Production target:

```bash
PRETRAIN_TARGET_TOKENS=1000000 PRETRAIN_TARGET_RUN=pretrain-prod-001 make pretrain-generate

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-prod-001
make pretrain-report PRETRAIN_REPORT_RUN=pretrain-prod-001
```

Push after inspection:

```bash
make pretrain-push HF_REPO=<namespace>/<repo>
```

Validated signal records are globally deduplicated and rendered into one public
file: `data/runs/<run>/deduped/pretrain.jsonl`. The public row contains `id`,
`text`, and `metadata.signal`; per-signal validated files remain internal run
artifacts and are not published as separate datasets.

Exact and five-token-shingle near-duplicate checks run across the complete
mixture, including across signal boundaries. Rejected duplicates are recorded
under `rejected/duplicates.jsonl`.

Pretraining completion is accepted-token based, not generated-row based. Each
round generates grounded candidates, validates them, rejects exact and
normalized near-duplicates globally, then counts estimated tokens from public
`text` only. Rejections create a token deficit and the next round consumes
unused candidate indexes for the affected signal. Signal shares therefore
apply to accepted text rather than attempted rows.

The pipeline stops successfully only after every selected signal reaches its
accepted-token target. If its genuinely distinct source inventory is exhausted,
or `generation.max_cost_usd` is reached, it preserves the accepted corpus,
writes `manifests/accepted_token_report.json`, and exits nonzero with the exact
shortfall. It never repeats or lightly renames records to fill the target.
Publishing also stops before creating a Hugging Face commit if a duplicate or
invalid public row remains.

Preflight scans the full finite inventory, ignores family labels when computing
structure fingerprints, and rejects a configuration whose estimated inventory
cannot cover its accepted-token allocation. Reporting and publishing separately
verify zero deficit for every configured signal.

## Validation Checklist

Generic alignment uses `generator -> deterministic validation -> judge ->
reviewer`. Models return only language-bearing fields through a minimal
plain-text contract; code owns IDs, metadata, taxonomy, tools, constraints, and
run fields. The judge must reject ambiguous or insufficiently grounded tasks
rather than guess. The reviewer sees only judge-accepted candidates and answers
whether the acceptance was justified. Final acceptance requires all three
gates. Semantic failure is final, is not backfilled, and is never repaired
locally. Provider or malformed-output failures retain bounded retry. Generator,
judge, and reviewer models are independently configurable, and role telemetry
is stored in batch manifests.

For each run, inspect:

- public rows or pairs for schema, formatting, and obvious quality failures
- run manifest planning fields, accepted-target status, and telemetry
- retry counts, adaptive batch failures, request tokens, and aggregate request seconds
- public dataset directory hygiene
- coverage reports or dataset cards before publishing
- `manifests/diversity_report_deduped.json` for normalized template reuse,
  near-duplicate clusters, artifact-family concentration, and cross-signal
  exact-template overlap; this report is a required clean gate

## See Also

- `COMMANDS.md` for Make target reference.
- `GENERATION_FAMILIES.md` for supported families/signals and target distribution behavior.
- `DATASET_PURPOSE.md` for artifact families and public row contracts.
