from prompts.arithmetic import (
    ARITHMETIC_CANDIDATE_SCHEMA,
    ARITHMETIC_CANDIDATE_TASK,
    ARITHMETIC_RESPONSE_SCHEMA,
    ARITHMETIC_RESPONSE_TASK,
)
from prompts.wrapper import build_batched_prompt, build_response_prompt
from slm_synth.repair import repair_arithmetic
from slm_synth.sources.two_pass import attach_candidate_ids, order_responses


class ArithmeticGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1, diversity_context: str = "", response_llm=None):
        self.llm = llm
        self.response_llm = response_llm or llm
        self.batch_size = batch_size
        self.diversity_context = diversity_context

    def generate_batch(self):
        candidate_prompt = build_batched_prompt(
            schema=ARITHMETIC_CANDIDATE_SCHEMA,
            task_instruction=ARITHMETIC_CANDIDATE_TASK,
            batch_size=self.batch_size,
            diversity_context=self.diversity_context,
        )
        candidates = attach_candidate_ids(self.llm.generate_batch(candidate_prompt, self.batch_size))
        response_prompt = build_response_prompt(
            schema=ARITHMETIC_RESPONSE_SCHEMA,
            task_instruction=ARITHMETIC_RESPONSE_TASK,
            candidates=candidates,
        )
        responses = order_responses(
            self.response_llm.generate_batch(response_prompt, self.batch_size), self.batch_size
        )
        records = []
        for candidate, response in zip(candidates, responses):
            records.append(
                repair_arithmetic(
                    {
                        "type": "arithmetic",
                        "question": candidate.get("question", ""),
                        "steps": response.get("steps"),
                        "answer": response.get("answer"),
                        "verification_expression": response.get("verification_expression"),
                        "verification_answer": response.get("verification_answer"),
                    }
                )
            )
        return records
