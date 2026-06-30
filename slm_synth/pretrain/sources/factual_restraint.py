from prompts.factual_restraint import (
    FACTUAL_RESTRAINT_CANDIDATE_SCHEMA,
    FACTUAL_RESTRAINT_CANDIDATE_TASK,
    FACTUAL_RESTRAINT_RESPONSE_SCHEMA,
    FACTUAL_RESTRAINT_RESPONSE_TASK,
)
from prompts.wrapper import build_batched_prompt, build_response_prompt
from slm_synth.pretrain.repair import repair_factual_restraint
from slm_synth.pretrain.sources.two_pass import attach_candidate_ids, order_responses


class FactualRestraintGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1, diversity_context: str = "", response_llm=None):
        self.llm = llm
        self.response_llm = response_llm or llm
        self.batch_size = batch_size
        self.diversity_context = diversity_context

    def generate_batch(self):
        candidate_prompt = build_batched_prompt(
            schema=FACTUAL_RESTRAINT_CANDIDATE_SCHEMA,
            task_instruction=FACTUAL_RESTRAINT_CANDIDATE_TASK,
            batch_size=self.batch_size,
            diversity_context=self.diversity_context,
        )
        candidates = attach_candidate_ids(self.llm.generate_batch(candidate_prompt, self.batch_size))
        response_prompt = build_response_prompt(
            schema=FACTUAL_RESTRAINT_RESPONSE_SCHEMA,
            task_instruction=FACTUAL_RESTRAINT_RESPONSE_TASK,
            candidates=candidates,
        )
        responses = order_responses(
            self.response_llm.generate_batch(response_prompt, self.batch_size), self.batch_size
        )
        return [
            repair_factual_restraint(
                {
                    "type": "factual_restraint",
                    "question": candidate.get("question", ""),
                    "safe_answer": response.get("safe_answer", ""),
                }
            )
            for candidate, response in zip(candidates, responses)
        ]
