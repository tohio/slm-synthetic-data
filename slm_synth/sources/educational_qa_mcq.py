import json
from slm_synth.prompt_loader import load_prompt
from slm_synth.schemas import validate_educational_qa_mcq


class EducationalQAMCQGenerator:
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
        raw = self.llm.generate_one(self.build_prompt())
        obj = json.loads(raw)
        validate_educational_qa_mcq(obj)
        return obj
