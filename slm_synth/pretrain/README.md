# `slm_synth/pretrain`

Grounded synthetic pretraining generation for one consolidated pretraining dataset.

## Production Path

```text
grounded artifacts/rendering
-> deterministic validation
-> Gemma judge
-> Luna reviewer
-> final global exact dedup
-> accepted-token accounting/backfill
-> deduped/pretrain.jsonl
```

The five signals are internal components of this one dataset:

- `arithmetic`
- `task_code`
- `educational_qa_mcq_math`
- `educational_qa_mcq_general`
- `factual_restraint`

`pipeline.py` owns semantic-quality completion and post-review token accounting. `curate.py` remains an internal grounded generation/validation/backfill mechanism used by the production pipeline; it is not a separate public workflow.

## Public Commands

```bash
make pretrain-smoke
make pretrain-generate
make pretrain-inspect
make pretrain-report
make pretrain-push
```

Hugging Face publication uses the final consolidated `deduped/pretrain.jsonl` artifact only.
