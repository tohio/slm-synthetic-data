# slm_synth/sources/factual_restraint.py

from prompts.wrapper import build_prompt
from slm_synth.repair import repair_factual_restraint
from slm_synth.schemas import FACTUAL_RESTRAINT_SCHEMA


class FactualRestraintGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_prompt(
            schema=FACTUAL_RESTRAINT_SCHEMA,
            task_instruction="",
            prompt_name="factual_restraint"
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_factual_restraint(obj)
        return obj

