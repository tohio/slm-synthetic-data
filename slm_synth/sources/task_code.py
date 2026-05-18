# slm_synth/sources/task_code.py

from slm_synth.prompts.task_code import build_task_code_prompt
from slm_synth.repair import repair_task_code


class TaskCodeGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_task_code_prompt()

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_task_code(obj)
        return obj
