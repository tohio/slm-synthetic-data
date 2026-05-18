from prompts.wrapper import build_batched_prompt
from slm_synth.schemas import EDUCATIONAL_QA_MCQ_SCHEMA
from prompts.educational_qa_mcq import EDUCATIONAL_QA_MCQ_TASK
from slm_synth.repair import repair_educational_qa_mcq

class EducationalQAMCQGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1):
        self.llm = llm
        self.batch_size = batch_size

    def generate_batch(self):
        prompt = build_batched_prompt(
            schema=EDUCATIONAL_QA_MCQ_SCHEMA,
            task_instruction=EDUCATIONAL_QA_MCQ_TASK,
            batch_size=self.batch_size,
        )
        objs = self.llm.generate_batch(prompt, self.batch_size)
        return [repair_educational_qa_mcq(o) for o in objs]
