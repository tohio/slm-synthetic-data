from prompts.wrapper import build_batched_prompt
from slm_synth.schemas import FACTUAL_RESTRAINT_SCHEMA
from prompts.factual_restraint import FACTUAL_RESTRAINT_TASK
from slm_synth.repair import repair_factual_restraint

class FactualRestraintGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1, diversity_context: str = ""):
        self.llm = llm
        self.batch_size = batch_size
        self.diversity_context = diversity_context

    def generate_batch(self):
        prompt = build_batched_prompt(
            schema=FACTUAL_RESTRAINT_SCHEMA,
            task_instruction=FACTUAL_RESTRAINT_TASK,
            batch_size=self.batch_size,
            diversity_context=self.diversity_context,
        )
        objs = self.llm.generate_batch(prompt, self.batch_size)
        return [repair_factual_restraint(o) for o in objs]
