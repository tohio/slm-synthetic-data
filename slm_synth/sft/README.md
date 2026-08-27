# `slm_synth/sft`

Generic supervised fine-tuning dataset generation.

## Production Path

```text
derivation
-> concrete task
-> exact/near task novelty filtering
-> answer generation
-> deterministic validation
-> Nemotron judge
-> Gemma reviewer
-> final exact dedup
```

`pipeline.py` owns generation. Shared runtime mechanics come from `slm_synth/runtime`; SFT-specific schemas, specifications, acceptance, reporting, and publication remain in this package.

## Task Families

- everyday conversation
- rewriting/editing
- summarization
- classification/extraction
- grounded QA/reading
- planning/brainstorming/recommendations
- creative writing
- programming
- applied math/reasoning
- safety/uncertainty/refusal

## Public Commands

```bash
make sft-smoke
make sft-generate
make sft-inspect
make sft-report
make sft-push
```

Publication remains one consolidated SFT repository with family JSONL files/configurations.
