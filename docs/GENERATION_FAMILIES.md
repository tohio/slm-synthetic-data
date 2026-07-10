# Generation Families

Supported family and signal names for each generation surface.

Use this file when planning run targets, interpreting coverage reports, or choosing a subset with `*_FAMILIES` / `*_SIGNALS` variables. Source files listed below are the implementation source of truth; update this document when those constants change.

## Selection Rules

| Surface | Selector | `all` behavior | Target distribution |
|---|---|---|---|
| Pretraining | `PRETRAIN_SIGNAL` | No value means all configured pretraining signals from `configs/synthetic_template.yaml`. | Token target is split by configured signal shares. |
| SFT | `SFT_FAMILIES` | `all` means all supported SFT spec families, sorted by name. | Row target is split evenly across families; remainder goes to earlier sorted families. |
| DPO | `DPO_FAMILIES` | `all` means all supported DPO spec families, sorted by name. | Pair target is split evenly across families; remainder goes to earlier sorted families. |
| Distillation SFT | `DISTILLATION_SFT_SIGNALS` | Empty or `all` means all supported distillation SFT signals, sorted by name. | Row target is split evenly across signals; remainder goes to earlier sorted signals. |
| Distillation DPO | `DISTILLATION_DPO_FAMILIES` | `all` means all supported distillation DPO families, sorted by name. | Pair target is split evenly across families; remainder goes to earlier sorted families. |

For SFT, DPO, distillation SFT, and distillation DPO, targets count accepted public rows or pairs. Rejected quality-gate outputs do not count toward the target.

## Pretraining Signals and Artifact Families

Pretraining uses grounded local artifacts that are rendered by the provider into public text records. Signal mix shares are configured in `configs/synthetic_template.yaml`.

| Signal | Configured share | Artifact families | Purpose |
|---|---:|---|---|
| `arithmetic` | 14.7540984% | `direct_expression`, `missing_operand`, `two_step_remaining_quantity`, `exact_allocation`, `unique_numeric_comparison` | Verified integer arithmetic and numeric reasoning text. |
| `task_code` | 39.3442623% | `normalized_counting`, `filter_sort_projection`, `grouped_totals`, `grouped_average_threshold`, `paired_comparison_counts`, `nested_transform`, `selection_by_total`, `dictionary_keywise_sum` | Short Python task/code behavior records. |
| `educational_qa_mcq_math` | 14.7540984% | `integer_expression`, `missing_operand`, `exact_division`, `two_step_quantity`, `unique_numeric_comparison` | Verified math multiple-choice questions. |
| `educational_qa_mcq_general` | 24.5901639% | `python_behavior`, `grammar`, `vocabulary`, `reading`, `fictional_rule`, `policy`, `scientific_method`, `ordering`, `final_location`, `table_lookup`, `threshold_rule`, `temporal_order`, `direction_following`, `conditional_access`, `comparison_claim`, `category_rule`, `cause_inference`, `schedule_availability`, `inventory_shortage`, `source_attribution`, `procedure_step`, `exception_rule`, `trend_interpretation`, `revision_tracking` | Grounded educational multiple-choice questions from supplied evidence. |
| `factual_restraint` | 6.5573770% | `future_uncertainty`, `ambiguous_entity`, `private_information`, `unannounced_information`, `rumor`, `medical`, `legal`, `financial` | Cautious-answer behavior for uncertainty, privacy, and high-stakes domains. |

Implementation source of truth:

```text
configs/synthetic_template.yaml
slm_synth/pretrain/artifacts/arithmetic.py
slm_synth/pretrain/artifacts/task_code.py
slm_synth/pretrain/artifacts/educational_qa_mcq_math.py
slm_synth/pretrain/artifacts/educational_qa_mcq_general.py
slm_synth/pretrain/artifacts/factual_restraint.py
```

## SFT Families

SFT families define eval-shaped supervised tasks. Public rows contain a user message, an assistant response, and metadata with `category`, `difficulty`, `template_family`, and `eval_family`.

| Family | Category | Template family | Purpose |
|---|---|---|---|
| `ai_concept_explanation` | `general_instruction_following` | `short_ai_definition` | Short machine-learning concept explanations. |
| `basic_arithmetic_qa` | `direct_arithmetic` | `direct_addition` | Direct addition with answer-only numeric responses. |
| `capital_city_qa` | `concise_factual_qa` | `capital_city_direct` | Capital-city factual answers with answer-only formatting. |
| `clear_sky_color_qa` | `concise_factual_qa` | `common_fact_color` | Common color facts and short factual answers. |
| `code_explanation_no_code` | `general_instruction_following` | `code_explanation_plain_text` | Plain-text explanation of small code snippets without fenced code. |
| `code_expression_result` | `code_expression_evaluation` | `python_expression_result` | Python expression evaluation with answer-only result. |
| `code_generation_function` | `code_generation` | `python_function_code_only` | Complete Python function generation, code only. |
| `direct_division` | `direct_arithmetic` | `direct_division` | Exact integer division with answer-only numeric responses. |
| `direct_subtraction` | `direct_arithmetic` | `direct_subtraction` | Direct subtraction with answer-only numeric responses. |
| `function_completion_body_only` | `code_generation` | `python_function_body_only` | Function body completion without repeating the signature. |
| `list_exact_n_items` | `exact_output_format_control` | `list_exact_count` | Exact-count list formatting. |
| `private_or_unverifiable_company_fact` | `private_info_restraint` | `private_company_metric` | Refusal/restraint for private or unverifiable company metrics. |
| `repeat_exact_n_times` | `exact_output_format_control` | `repeat_word_count` | Exact repeat-count output control. |
| `short_factual_stop_behavior` | `controlled_verbosity` | `short_factual_answer` | Short factual answers that stop when complete. |

Implementation source of truth:

```text
slm_synth/sft/spec_builders.py
```

## DPO Families

Generic DPO uses the same supported family set as SFT, but each family adds a rejected-response failure mode. Public pairs contain `prompt`, `chosen`, `rejected`, and metadata with `failure_mode`.

| Family | Failure mode | Preference objective |
|---|---|---|
| `ai_concept_explanation` | `wrong_factual_answer` | Prefer correct, concise concept explanation over incorrect explanation. |
| `basic_arithmetic_qa` | `wrong_numeric_answer` | Prefer exact numeric answer over wrong number. |
| `capital_city_qa` | `wrong_factual_answer` | Prefer correct capital over incorrect factual answer. |
| `clear_sky_color_qa` | `wrong_factual_answer` | Prefer correct common fact over incorrect answer. |
| `code_explanation_no_code` | `code_includes_explanation` | Prefer compliant plain-text explanation over response that violates the expected surface. |
| `code_expression_result` | `wrong_numeric_answer` | Prefer exact expression result over wrong value. |
| `code_generation_function` | `code_includes_explanation` | Prefer code-only complete function over Markdown/prose-wrapped response. |
| `direct_division` | `wrong_numeric_answer` | Prefer exact integer quotient over wrong number. |
| `direct_subtraction` | `wrong_numeric_answer` | Prefer exact subtraction result over wrong number. |
| `function_completion_body_only` | `code_includes_explanation` | Prefer body-only function completion over response with prose/signature leakage. |
| `list_exact_n_items` | `format_violation` | Prefer exact item count and separators over extra/misformatted items. |
| `private_or_unverifiable_company_fact` | `unknown_fact_fabrication` | Prefer restraint over fabricated private or unverifiable details. |
| `repeat_exact_n_times` | `format_violation` | Prefer exact repeat count over extra/missing repeated items. |
| `short_factual_stop_behavior` | `verbosity_mismatch` | Prefer concise answer-only response over verbose completion. |

Implementation source of truth:

```text
slm_synth/dpo/spec_builders.py
slm_synth/sft/spec_builders.py
```

## Distillation SFT Signals

Distillation SFT creates teacher prompt/response rows. Public rows are per-signal JSONL files under `data/distillation/runs/<run>/datasets/`.

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
