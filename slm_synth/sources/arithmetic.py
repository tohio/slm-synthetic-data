from prompts.wrapper import build_batched_prompt
from slm_synth.schemas import ARITHMETIC_SCHEMA
from prompts.arithmetic import ARITHMETIC_TASK
from slm_synth.repair import repair_arithmetic

class ArithmeticGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1):
        self.llm = llm
        self.batch_size = batch_size

    def generate_batch(self):
        prompt = build_batched_prompt(
            schema=ARITHMETIC_SCHEMA,
            task_instruction=ARITHMETIC_TASK,
            batch_size=self.batch_size,
        )
        objs = self.llm.generate_batch(prompt, self.batch_size)
        return [repair_arithmetic(o) for o in objs]
