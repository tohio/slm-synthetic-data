from prompts.wrapper import build_prompt
from slm_synth.repair import repair_task_code
from slm_synth.schemas import TASK_CODE_SCHEMA
from slm_synth.prompts.task_code import TASK_CODE_TASK


class TaskCodeGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_prompt(
            schema=TASK_CODE_SCHEMA,
            task_instruction=TASK_CODE_TASK,
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_task_code(obj)
        return obj
