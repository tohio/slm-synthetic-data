# `slm_synth/distillation_dpo`

Preference-pair generation for post-distillation DPO alignment.

## Production Path

```text
derivation
-> task
-> task novelty
-> chosen/rejected pair generation
-> deterministic pair validation
-> five-gate Gemma judge
-> Luna reviewer
-> final exact dedup
```

The judge requires all five gates:

- assessable
- chosen complete
- chosen correct
- preference valid
- dimension aligned

Distillation DPO uses distillation-specific derivation/task guidance, rejected-pair defect guidance, and reviewer calibration. It must remain semantically separate from generic DPO.

## Public Commands

```bash
make distillation-dpo-smoke
make distillation-dpo-generate
make distillation-dpo-inspect
make distillation-dpo-report
make distillation-dpo-push
```

The public product is one consolidated Distillation-DPO repository containing `teacher_response_preference.jsonl`.
