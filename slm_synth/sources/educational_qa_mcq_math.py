from prompts.educational_qa_mcq_math import (
    EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_SCHEMA,
    EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_TASK,
    EDUCATIONAL_QA_MCQ_MATH_RESPONSE_SCHEMA,
    EDUCATIONAL_QA_MCQ_MATH_RESPONSE_TASK,
)
from prompts.wrapper import build_batched_prompt, build_response_prompt
from slm_synth.repair import repair_educational_qa_mcq_math
from slm_synth.sources.two_pass import attach_candidate_ids, order_responses


class EducationalQAMCQMathGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1, diversity_context: str = "", response_llm=None):
        self.llm = llm
        self.response_llm = response_llm or llm
        self.batch_size = batch_size
        self.diversity_context = diversity_context

    def generate_batch(self):
        candidate_prompt = build_batched_prompt(
            schema=EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_SCHEMA,
            task_instruction=EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_TASK,
            batch_size=self.batch_size,
            diversity_context=self.diversity_context,
        )
        candidates = attach_candidate_ids(self.llm.generate_batch(candidate_prompt, self.batch_size))
        response_prompt = build_response_prompt(
            schema=EDUCATIONAL_QA_MCQ_MATH_RESPONSE_SCHEMA,
            task_instruction=EDUCATIONAL_QA_MCQ_MATH_RESPONSE_TASK,
            candidates=candidates,
        )
        responses = order_responses(
            self.response_llm.generate_batch(response_prompt, self.batch_size), self.batch_size
        )
        records = []
        for candidate, response in zip(candidates, responses):
            records.append(
                repair_educational_qa_mcq_math(
                    {
                        "type": "educational_qa_mcq_math",
                        "question": response.get("question", candidate.get("question", "")),
                        "choices": response.get("choices", candidate.get("choices", [])),
                        "correct_index": response.get("correct_index"),
                        "explanation": response.get("explanation", ""),
                        "verification_expression": response.get("verification_expression", ""),
                        "verification_answer": response.get("verification_answer", ""),
                    }
                )
            )
        return records
