from prompts.wrapper import build_batched_prompt
from slm_synth.schemas import TASK_CODE_SCHEMA
from prompts.task_code import TASK_CODE_TASK
from slm_synth.repair import repair_task_code

class TaskCodeGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1):
        self.llm = llm
        self.batch_size = batch_size

    def generate_batch(self):
        prompt = build_batched_prompt(
            schema=TASK_CODE_SCHEMA,
            task_instruction=TASK_CODE_TASK,
            batch_size=self.batch_size,
        )
        objs = self.llm.generate_batch(prompt, self.batch_size)
        return [repair_task_code(o) for o in objs]
