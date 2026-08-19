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

Smoke:

```bash
make sft-smoke
make sft-inspect SFT_INSPECT_RUN=sft-smoke-001
```

Small candidate run:

```bash
SFT_FAMILIES="basic_arithmetic_qa ai_concept_explanation" \
SFT_CANDIDATE_COUNTS="basic_arithmetic_qa=4 ai_concept_explanation=2" \
SFT_GENERATION_RUN=sft-small-001 \
make sft-generate

make sft-inspect SFT_INSPECT_RUN=sft-small-001
make sft-report SFT_REPORT_RUN=sft-small-001
```

Another explicit candidate plan (choose counts only after reviewing family capacity and pilot quality):

```bash
SFT_FAMILIES="basic_arithmetic_qa ai_concept_explanation" \
SFT_CANDIDATE_COUNTS="basic_arithmetic_qa=8 ai_concept_explanation=4" \
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

### SFT reporting and publish blockers

`make sft-report` loads `configs/eval_holdouts.yaml` by default and writes `coverage.json`. The report includes aggregate and per-family metadata coverage, normalized ID/prompt/conversation uniqueness, response repetition, holdout collisions, and candidate/attempted/accepted/rejected/duplicate counts.

Publication is blocked when:

- IDs, normalized prompts, or normalized conversations repeat.
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
│   ├── ai_concept_explanation.jsonl
│   ├── basic_arithmetic_qa.jsonl
│   └── ...
└── artifacts/
    ├── coverage.json
    └── manifests/
```

The `default` configuration loads all `data/*.jsonl` files as the train split. Each family configuration loads its existing family file; it does not duplicate stored rows. Families remain metadata/configuration boundaries, not train/validation/test splits.

The push target does not modify or delete legacy `slm-synthetic-sft-*` family repositories. Keep those repositories until the consolidated dataset has been published, loaded with both default and family configurations, inspected, and adopted by downstream consumers.

## Generic DPO

Smoke:

```bash
make dpo-smoke
make dpo-inspect DPO_INSPECT_RUN=dpo-smoke-001
```

Small candidate run:

```bash
DPO_TARGET_PAIRS=100 DPO_TARGET_RUN=dpo-small-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-small-001
make dpo-report DPO_REPORT_RUN=dpo-small-001
```

Production target:

```bash
DPO_TARGET_PAIRS=14000 DPO_TARGET_RUN=dpo-prod-001 make dpo-generate

make dpo-inspect DPO_INSPECT_RUN=dpo-prod-001
make dpo-report DPO_REPORT_RUN=dpo-prod-001
```

Push after inspection:

```bash
make dpo-push \
  DPO_PUSH_RUN=dpo-prod-001 \
  DPO_HF_REPO=tohio/slm-synthetic-dpo
```

Public pairs are written under `data/dpo/runs/<run>/datasets/`, with one final JSONL file per family. Batch shards remain under the sibling `batches/` directory.

### DPO acceptance and backfill

DPO accepts the first pair for each normalized ID, prompt, and complete `(prompt, chosen, rejected)` triple. Duplicate pairs and terminal local-validation failures do not count toward the target. Chosen/rejected similarity and repeated negative patterns are reported but remain diagnostic because controlled near-miss negatives can be intentional.

Each generation round uses the next unused source indexes. If the run remains underfilled after its configured budget, generation preserves accepted public pairs, writes a failed run manifest, and exits nonzero. `DPO_MAX_BACKFILL_ROUNDS` is the total lifetime budget, including rounds recorded before resume.

Resume a finalized underfilled run by keeping its run id and source plan, increasing the total budget, and enabling resume:

```bash
DPO_TARGET_PAIRS=14000 \
DPO_TARGET_RUN=dpo-prod-001 \
DPO_MAX_BACKFILL_ROUNDS=3 \
DPO_RESUME=true \
make dpo-generate
```

Resume validates the public files, content fingerprints, family allocation, batch manifests, accepted-target accounting, and next source indexes before making a request. A complete run resumes as a no-op.

### DPO reporting and publish blockers

`make dpo-report` loads `configs/eval_holdouts.yaml` and writes `coverage.json`. The report includes aggregate and per-family metadata coverage, normalized ID/prompt/triple uniqueness, chosen/rejected similarity, negative-pattern distribution, holdout collisions, and attempted/accepted/rejected/duplicate/remaining counts.

Publication is blocked when:

- IDs, normalized prompts, or normalized preference triples repeat.
- Holdouts were not checked or a collision exists.
- Accepted-target accounting is missing, inconsistent, or underfilled.
- `coverage.json` is missing or stale relative to the public files.
- The dataset card lacks the default or any required family configuration.

### DPO consolidated publication

`dpo-push` publishes one atomic commit to `DPO_HF_REPO`:

```text
tohio/slm-synthetic-dpo/
├── README.md
├── data/
│   ├── ai_concept_explanation.jsonl
│   ├── basic_arithmetic_qa.jsonl
│   └── ...
└── artifacts/
    ├── coverage.json
    └── manifests/
```

The `default` configuration loads all `data/*.jsonl` files as the train split. Each family configuration loads its family file without duplicating stored pairs. Families are metadata/configuration boundaries, not train/validation/test splits.

The push target does not modify legacy `slm-synthetic-dpo-*` repositories. Keep them until the consolidated dataset has been published, loaded through both default and family configurations, inspected, and adopted downstream.

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
under `rejected/duplicates.jsonl`. There is no backfill to restore a requested
row count, and publishing stops before creating a Hugging Face commit if a
duplicate or invalid public row remains.

## Validation Checklist

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
