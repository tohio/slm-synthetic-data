# Generation Families

Supported family and signal names for each generation surface.

Use this file when planning run targets, interpreting coverage reports, or choosing a subset with `*_FAMILIES` / `*_SIGNALS` variables. Source files listed below are the implementation source of truth; update this document when those constants change.

## Selection Rules

| Surface | Selector | `all` behavior | Generation planning |
|---|---|---|---|
| Pretraining | `PRETRAIN_SIGNAL` | No value means all configured pretraining signals from `configs/synthetic_template.yaml`. | Token target is split by configured signal shares. |
| SFT | `SFT_FAMILIES` | `all` means all supported SFT spec families, sorted by name. | `SFT_CANDIDATE_COUNTS` must explicitly assign a candidate count to every selected family. |
| DPO | `DPO_FAMILIES` | `all` means all supported DPO preference dimensions, sorted by name. | `DPO_CANDIDATE_COUNTS` must explicitly assign a candidate count to every selected dimension. |
| Distillation SFT | `DISTILLATION_SFT_SIGNALS` | Empty or `all` means all supported distillation SFT signals, sorted by name. | `DISTILLATION_SFT_CANDIDATE_COUNTS` must explicitly assign a candidate count to every selected signal. |
| Distillation DPO | `DISTILLATION_DPO_FAMILIES` | `all` means all supported distillation DPO families, sorted by name. | Pair target is split evenly across families; remainder goes to earlier sorted families. |

For generic SFT, generic DPO, and distillation SFT, accepted outputs are outcomes: rejected or duplicate candidates are not replaced to fill a quota.

## Pretraining Signals and Artifact Families

Pretraining uses grounded local artifacts that are rendered by the provider into text records. Signal mix shares are configured in `configs/synthetic_template.yaml`. Validated signals are globally exact/near deduplicated and consolidated into one public `pretrain.jsonl`; signal identity remains available as `metadata.signal` rather than separate published datasets.

| Signal | Configured share | Artifact families | Purpose |
|---|---:|---|---|
| `arithmetic` | 14.7540984% | `direct_expression`, `missing_start_after_increase`, `missing_start_after_decrease`, `two_step_remaining`, `gain_then_spend`, `exact_group_count`, `equal_share_size`, `groups_with_loose_items`, `compare_group_totals`, `target_gap`, `three_source_total`, `constant_rate_total`, `two_rate_total`, `known_portion_equal_shares`, `net_change`, `rectangle_perimeter` | Verified integer arithmetic spread across distinct inverse, change, grouping, rate, comparison, sharing, and measurement relationships. Semantic contexts rotate before operand variants, and preflight rejects a planned arithmetic run if source structures repeat. |
| `task_code` | 39.3442623% | 200 materially distinct single-function Python tasks spanning data processing, algorithms, data structures, graphs, strings, numerical methods, geometry, scheduling, and systems simulations. | Validated local code and deterministic tasks with teacher-generated implementation plans. |
| `educational_qa_mcq_math` | 14.7540984% | 24 verified relationships rendered in 11 semantically grounded application contexts, for 264 finite candidates. | Locally authoritative math multiple-choice questions with teacher-generated explanations only. |
| `educational_qa_mcq_general` | 24.5901639% | 24 reasoning families expressed through 9 evidence-document forms, for 216 finite candidates. | Locally authoritative evidence-grounded multiple-choice questions with teacher-generated explanations only. |
| `factual_restraint` | 6.5573770% | 32 restraint behaviors with 4 materially different scenarios each, for 128 finite candidates. | Calibrated restraint without fabricated facts, unsafe disclosure, or unsupported high-stakes decisions. |

Arithmetic has a declared capacity of 288 structurally distinct candidates.
Token planning treats that as a ceiling rather than cycling through operand-only
variants. Larger total-token requests therefore do not silently manufacture
additional arithmetic rows from the same structures.

`task_code` has a declared capacity of 200 distinct tasks. Each task
appears once; field-name substitutions, threshold changes, and renamed copies
are not treated as additional training data. The public task is deterministic
from the validated local artifact, while the teacher supplies only its concise
implementation plan.

`educational_qa_mcq_math` has a declared capacity of 264 locally verified
questions: 24 mathematical relationships across 11 application contexts.
Numbers and contextual facts change together; number-only copies are not
counted. The provider cannot rewrite questions, choices, or answers and
supplies only an explanation of the verified calculation.

`educational_qa_mcq_general` has a declared capacity of 216 candidates. Each of
24 reasoning families appears in nine meaningfully different evidence-document
forms. Evidence, questions, choices, and answers remain local, while the
teacher supplies only the explanation. Entity-only substitutions do not create
additional document forms.

`factual_restraint` has a declared capacity of 128 materially distinct
scenarios: four applications of each of 32 restraint behaviors. Each source
question and behavior requirement is local; the teacher supplies only the
user-facing answer. The scenarios change the underlying case, not merely an
entity, date, location, or amount.

Implementation source of truth:

```text
configs/synthetic_template.yaml
slm_synth/pretrain/artifacts/arithmetic.py
slm_synth/pretrain/artifacts/task_code.py
slm_synth/pretrain/artifacts/task_code_catalog.py
slm_synth/pretrain/artifacts/educational_qa_mcq_math.py
slm_synth/pretrain/artifacts/educational_qa_mcq_general.py
slm_synth/pretrain/artifacts/factual_restraint.py
```

## SFT Families

Generic SFT uses ten broad task families rather than eval-shaped families:

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

Coverage is also labeled by `interaction_modes`, `output_mode`, and `context_mode`. Generic DPO uses the same task axes plus one of ten preference dimensions: helpfulness and completeness, factual accuracy, instruction adherence, appropriate detail, organization, style and tone, tool-call correctness, groundedness, safe-refusal calibration, or code correctness. `eval_family` is not part of generic SFT/DPO specs or public artifacts.

The SFT source inventory is finite and manually curated: six materially
different briefs per task family, for 60 declared candidates. Each source has
an internal semantic `source_key`; capacity is the actual catalog length, not
a template multiplier. The inventory intentionally covers single-turn,
multi-turn, system-conditioned, concise, structured, tabular, constrained,
code, supplied-passage, multi-document, and long-document tasks.

Implementation source of truth:

```text
slm_synth/sft/spec_builders.py
```

## DPO Preference Dimensions

Public pairs contain one shared `prompt`, explicit multi-message `chosen` and
`rejected` branches, optional shared `tools`, the shared SFT axes, one
`preference_dimension`, and a concrete `failure_mode`. Tool-use branches are
validated against the same prompt and tool inventory.

| Preference dimension | Representative failure mode | Preference objective |
|---|---|---|
| `helpfulness_and_completeness` | `incomplete_response` | Prefer a response that fully addresses the request. |
| `factual_accuracy` | `unsupported_claim` | Prefer correct, supportable claims. |
| `instruction_adherence` | `instruction_violation` | Prefer compliance with explicit constraints. |
| `appropriate_detail` | `excessive_detail` | Prefer detail calibrated to the request. |
| `organization` | `poor_organization` | Prefer coherent, usable structure. |
| `style_and_tone` | `tone_mismatch` | Prefer the requested tone and audience fit. |
| `tool_call_correctness` | `incorrect_tool_call` | Prefer correct tool selection and arguments. |
| `groundedness` | `ungrounded_response` | Prefer answers supported by supplied context. |
| `safe_refusal_calibration` | `over_refusal` | Prefer safe help without unnecessary refusal. |
| `code_correctness` | `code_logic_error` | Prefer code that satisfies the stated behavior. |

Implementation source of truth:

```text
slm_synth/dpo/spec_builders.py
```

Every DPO preference dimension has a unique source capacity of nine
independently authored prompts, for 90 declared candidates. DPO does not
import, rename, or transform SFT source specs.
Generation preflights the full DPO inventory, its separation from SFT, and the
explicit candidate range before constructing a provider backend. Accepted
output must have unique normalized prompts and complete preference triples;
quality-rejected candidates are not replaced.

`make alignment-preflight` audits both complete inventories without contacting
a provider. It rejects duplicate semantic keys, exact or near-duplicate source
content, number-only variants, numbered variants, template concentration,
missing axis coverage, invalid metadata, and DPO prompts copied from SFT.

Failure-mode coverage, chosen/rejected similarity, and repeated negative constructions are reported at aggregate and per-dimension levels. They are inspection signals rather than automatic rejection thresholds. Exact duplicates, holdout collisions, empty output, and inconsistent accounting block publication.

## Distillation SFT Signals

Distillation SFT creates teacher prompt/response rows. Public rows are per-signal JSONL files under `data/distillation/runs/<run>/datasets/` and include filterable `category`, `difficulty`, `template_family`, and `eval_family` metadata.

| Signal | Purpose |
|---|---|
| `arithmetic` | Teacher responses for arithmetic and numeric reasoning prompts. |
| `cloud` | Cloud architecture, deployment, IAM, cost, and operational guidance prompts. |
| `code` | Code-writing and code-oriented instruction prompts. |
| `data_transform` | Data cleanup, transformation, mapping, and structured-output prompts. |
| `database` | Query, schema, indexing, and database operation prompts. |
| `debugging` | Diagnose and fix small code or operational issues. |
| `educational_qa` | General educational question-answering prompts. |
| `factual_restraint` | Safe restraint for private, unverifiable, future, or high-stakes questions. |
| `instruction` | General instruction-following prompts. |
| `planning` | Step plans, checklists, and practical task-planning prompts. |

Implementation source of truth:

```text
slm_synth/distillation_sft/signals.py
slm_synth/distillation_sft/seeds.py
slm_synth/distillation_sft/spec_builders.py
slm_synth/distillation_sft/public_metadata.py
```

## Distillation DPO Families

Distillation DPO is isolated from generic DPO. It is LLM-backed and uses deterministic preference specs as anchors for teacher-quality `chosen` responses and controlled-weak `rejected` responses.

| Family | Internal template coverage | Purpose |
|---|---|---|
| `teacher_response_preference` | arithmetic, answer-only factual QA, exact repeat/list formatting, code function generation, code expression evaluation, factual restraint, subtraction, division | Preference pairs for aligning distilled models toward teacher-quality answers and away from controlled failure modes. |

Implementation source of truth:

```text
slm_synth/distillation_dpo/seeds.py
slm_synth/distillation_dpo/spec_builders.py
slm_synth/distillation_dpo/runs.py
```

## See Also

- `GENERATION_WORKFLOW.md` for the run ladder.
- `COMMANDS.md` for Make targets and variables.
- `DATASET_PURPOSE.md` for public row schemas and metadata boundaries.
