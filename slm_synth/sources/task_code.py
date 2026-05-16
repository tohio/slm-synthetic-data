import json
from pathlib import Path
from slm_synth.schemas import validate_task_code
from slm_synth.prompt_loader import load_prompt


class TaskCodeGenerator:
    def __init__(self, llm, prompt_file: str):
        self.llm = llm
        self.prompt = load_prompt(prompt_file)

    def build_prompt(self):
        return (
            f"{self.prompt['system']}\n\n"
            f"Instruction:\n{self.prompt['instruction']}\n\n"
            f"Output format:\n{self.prompt['format']}"
        )

    def generate(self):
        prompt = self.build_prompt()
        raw = self.llm.generate(prompt)
        obj = json.loads(raw)
        validate_task_code(obj)
        return obj
