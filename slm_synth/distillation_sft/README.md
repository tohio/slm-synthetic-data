# `slm_synth/distillation_sft`

Teacher prompt/response generation for response distillation.

## Production Path

```text
derivation
-> student-appropriate prompt
-> task novelty
-> teacher response
-> deterministic row/response validation
-> response novelty
-> Nemotron judge
-> Gemma reviewer
-> final exact prompt/response dedup
```

The final dedup rejects duplicate prompt+response pairs, duplicate prompts, and duplicate responses. Reviewer calibration includes contradiction checks and code-family calibration.

## Signals

`arithmetic`, `cloud`, `code`, `data_transform`, `database`, `debugging`, `educational_qa`, `factual_restraint`, `instruction`, `planning`.

## Public Commands

```bash
make distillation-sft-smoke
make distillation-sft-generate
make distillation-sft-inspect
make distillation-sft-report
make distillation-sft-push
```

Manual post-run adjudication is not a supported path; judge and reviewer stages are part of production generation.
