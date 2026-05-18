import json
from slm_synth.prompt_loader import load_prompt
from slm_synth.schemas import validate_arithmetic


class ArithmeticGenerator:
    def __init__(self, llm, prompt_file: str):
        self.llm = llm
        self.prompt = load_prompt(prompt_file)

    def build_prompt(self):
        return (
            f"{self.prompt['system']}\n\n"
            f"{self.prompt['instruction']}\n\n"
            f"Output format:\n{self.prompt['format']}"
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())  # already a dict
        validate_arithmetic(obj)
        return obj

