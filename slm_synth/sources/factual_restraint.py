from prompts.wrapper import build_prompt
from slm_synth.repair import repair_factual_restraint
from slm_synth.schemas import FACTUAL_RESTRAINT_SCHEMA
from slm_synth.prompts.factual_restraint import FACTUAL_RESTRAINT_TASK


class FactualRestraintGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_prompt(
            schema=FACTUAL_RESTRAINT_SCHEMA,
            task_instruction=FACTUAL_RESTRAINT_TASK,
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_factual_restraint(obj)
        return obj
