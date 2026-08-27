# Generation Families

## Generic SFT Task Families

Generic SFT supports ten broad task families:

- `everyday_conversation`
- `rewriting_and_editing`
- `summarization`
- `classification_and_extraction`
- `grounded_qa_and_reading`
- `planning_brainstorming_recommendations`
- `creative_writing`
- `programming`
- `applied_math_and_reasoning`
- `safety_uncertainty_and_refusal`

Each selected family follows the same production stages:

```text
derivation -> task -> novelty -> answer -> deterministic validation
-> judge -> reviewer -> final dedup
```

## Generic DPO Preference Dimensions

Generic DPO supports ten semantic preference dimensions:

- `helpfulness_and_completeness`
- `factual_accuracy`
- `instruction_adherence`
- `detail`
- `organization`
- `style_and_tone`
- `tool_call_correctness`
- `groundedness`
- `safe_refusal`
- `code_correctness`

The model-facing pipeline uses plain-text prompt/chosen/rejected semantics. Final accepted rows are adapted to the repository's public DPO message schema for reporting and publication.

## Distillation SFT Signals

Distillation SFT supports ten signals:

- `arithmetic`
- `cloud`
- `code`
- `data_transform`
- `database`
- `debugging`
- `educational_qa`
- `factual_restraint`
- `instruction`
- `planning`

All signals share the production path:

```text
derivation -> student-appropriate prompt -> task novelty -> teacher response
-> deterministic validation -> response novelty -> judge -> reviewer
-> final prompt/response dedup
```

## Distillation DPO Preference Dimensions

Distillation DPO uses the same ten semantic dimension names as generic DPO, but its prompts, pair-generation guidance, judge gates, and reviewer calibration are distillation-specific.

The judge accepts only when all five gates pass:

- `assessable`
- `chosen_complete`
- `chosen_correct`
- `preference_valid`
- `dimension_aligned`

Do not collapse Distillation DPO semantics into generic DPO.

## Pretraining Signals

Pretraining uses five deterministic grounded signal families:

- `arithmetic`
- `task_code`
- `educational_qa_mcq_math`
- `educational_qa_mcq_general`
- `factual_restraint`

They are internal components of one consolidated pretraining dataset.
