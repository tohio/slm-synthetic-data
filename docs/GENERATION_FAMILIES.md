# Generation Families and Dimensions

Coverage reference for the internal signals, task families, and preference dimensions that make up the five public dataset products.

## Pretraining Signals

The five pretraining signals are internal components of one consolidated dataset.

| Signal | Intended capability | Typical deterministic grounding |
|---|---|---|
| `arithmetic` | exact numerical calculation and compact reasoning | locally generated arithmetic artifacts with verifiable answers |
| `task_code` | Python task understanding, plan, and implementation | deterministic task catalog/artifacts |
| `educational_qa_mcq_math` | numerical multiple-choice reasoning | locally verifiable math questions/answer keys |
| `educational_qa_mcq_general` | evidence-grounded educational QA | self-contained rules/passages/code snippets |
| `factual_restraint` | calibrated uncertainty and non-invention | prompts whose missing/private/future information is known by construction |

These are **signals**, not five Hugging Face products. They are globally combined and deduplicated into the one pretraining dataset.

## Generic SFT Families

| Family | Coverage |
|---|---|
| `everyday_conversation` | natural conversational help, follow-up, tone, and context |
| `rewriting_and_editing` | rewrite, polish, transform, preserve meaning/constraints |
| `summarization` | concise and structured summarization |
| `classification_and_extraction` | labels, extraction, structured fields |
| `grounded_qa_and_reading` | reading comprehension and answers grounded in supplied context |
| `planning_brainstorming_recommendations` | practical plans, options, recommendations, trade-offs |
| `creative_writing` | controlled creative generation |
| `programming` | Python/code tasks and implementation explanations |
| `applied_math_and_reasoning` | multi-step quantitative and logical tasks |
| `safety_uncertainty_and_refusal` | appropriate uncertainty, refusal, and safe redirection |

A family may exercise multiple interaction modes such as system conditioning, multi-turn conversation, structured output, or tool-mediated behavior when the SFT specification requires it.

## Generic DPO Preference Dimensions

| Dimension | What the chosen response should improve |
|---|---|
| `helpfulness_and_completeness` | answer the full request with material useful content |
| `factual_accuracy` | avoid false statements and incorrect conclusions |
| `instruction_adherence` | obey explicit constraints and requested format |
| `detail` | provide the appropriate amount of useful detail |
| `organization` | improve structure and navigability |
| `style_and_tone` | match requested tone/style without degrading content |
| `tool_call_correctness` | choose/call tools with correct arguments and interpretation |
| `groundedness` | stay supported by supplied evidence/context |
| `safe_refusal` | refuse only when appropriate and remain helpful |
| `code_correctness` | produce code that actually satisfies the requested behavior |

The pair generator should create a **material** preference, not a cosmetic rewrite. Judge/reviewer stages reject pairs where chosen and rejected are effectively equivalent.

## Distillation SFT Signals

| Signal | Intended teacher-response coverage |
|---|---|
| `arithmetic` | quantitative problem solving and explanation |
| `cloud` | cloud infrastructure, operations, architecture, and troubleshooting |
| `code` | implementation and code reasoning |
| `data_transform` | parsing, reshaping, normalization, and transformation tasks |
| `database` | SQL/database design, querying, and troubleshooting |
| `debugging` | diagnose defects and propose correct fixes |
| `educational_qa` | clear explanatory answers for learning tasks |
| `factual_restraint` | uncertainty, unverifiable/private facts, non-invention |
| `instruction` | precise instruction following |
| `planning` | structured planning and multi-step execution |

Distillation SFT task generation is explicitly student-appropriate: the prompt must contain enough information to answer without relying on unavailable attachments or hidden context.

## Distillation DPO Dimensions

Distillation DPO uses the same ten dimension names as generic DPO:

~~~text
helpfulness_and_completeness
factual_accuracy
instruction_adherence
detail
organization
style_and_tone
tool_call_correctness
groundedness
safe_refusal
code_correctness
~~~

The semantics are **not** interchangeable with generic DPO. Distillation DPO has:

- distillation-specific derivation guidance;
- dimension-specific weak-pair defect guidance;
- a stricter five-gate judge;
- Luna reviewer calibration;
- one consolidated `teacher_response_preference.jsonl` public product.

## Diversity Expectations

The generation target is not “maximum surface variation at any cost.” Useful diversity should preserve task correctness and semantic intent while avoiding template collapse.

The runtime uses exact and near-duplicate detection to reject high-similarity tasks before expensive answer/pair stages. Dataset-specific final dedup then enforces the appropriate public uniqueness rule.

## See Also

- [Dataset Purpose and Contracts](DATASET_PURPOSE.md)
- [Generation Workflow](GENERATION_WORKFLOW.md)
