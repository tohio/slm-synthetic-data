from prompts.wrapper import build_prompt
from slm_synth.repair import repair_educational_qa_mcq
from slm_synth.schemas import EDUCATIONAL_QA_MCQ_SCHEMA
from slm_synth.prompts.educational_qa_mcq import EDUCATIONAL_QA_MCQ_TASK


class EducationalQAMCQGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_prompt(
            schema=EDUCATIONAL_QA_MCQ_SCHEMA,
            task_instruction=EDUCATIONAL_QA_MCQ_TASK,
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_educational_qa_mcq(obj)
        return obj
