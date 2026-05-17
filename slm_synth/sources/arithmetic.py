import json
from slm_synth.schemas import validate_arithmetic
from slm_synth.prompt_loader import load_prompt


class ArithmeticGenerator:
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
        raw = self.llm.generate(self.build_prompt())
        obj = json.loads(raw)
        validate_arithmetic(obj)
        return obj

    def build_batched_prompt(self, batch_size: int):
        schema = self.prompt["format"]
        instruction = self.prompt["instruction"]

        header = (
            f"You are a data generator. Produce EXACTLY {batch_size} independent samples.\n\n"
            f"Each sample must follow this JSON schema:\n\n"
            f"{schema}\n\n"
            f"Return ONLY a JSON array of length {batch_size}.\n"
            f"No explanations. No prose. No comments.\n\n"
        )

        blocks = [
            f"### SAMPLE {i+1} INSTRUCTION\n{instruction}\n"
            for i in range(batch_size)
        ]

        return header + "\n".join(blocks)

    def generate_batch(self, batch_size: int):
        arr = self.llm.generate(
            self.build_batched_prompt(batch_size),
            expect_array=True,
            expected_length=batch_size
        )

        validated = []
        for obj in arr:
            validate_arithmetic(obj)
            validated.append(obj)

        return validated
