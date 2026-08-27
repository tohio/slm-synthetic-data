# `slm_synth/dpo`

Generic preference dataset generation for DPO alignment.

## Production Path

```text
derivation
-> concrete task
-> exact/near task novelty filtering
-> chosen/rejected pair generation
-> deterministic pair validation
-> Nemotron judge
-> Gemma reviewer
-> final exact preference-triple dedup
```

The model-facing pipeline uses plain-text prompt/chosen/rejected semantics. Final accepted rows are adapted to the repository public DPO message schema.

## Preference Dimensions

- helpfulness and completeness
- factual accuracy
- instruction adherence
- detail
- organization
- style and tone
- tool-call correctness
- groundedness
- safe refusal
- code correctness

## Public Commands

```bash
make dpo-smoke
make dpo-generate
make dpo-inspect
make dpo-report
make dpo-push
```

Publication remains one consolidated DPO repository with dimension JSONL files/configurations.
