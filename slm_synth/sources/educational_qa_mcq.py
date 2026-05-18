# slm_synth/sources/educational_qa_mcq.py

from prompts.wrapper import build_prompt
from slm_synth.repair import repair_educational_qa_mcq


class EducationalQAMCQGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_educational_qa_mcq_prompt()

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_educational_qa_mcq(obj)
        return obj
