from prompts.wrapper import build_batched_prompt
from slm_synth.schemas import EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA
from prompts.educational_qa_mcq_general import EDUCATIONAL_QA_MCQ_GENERAL_TASK
from slm_synth.repair import repair_educational_qa_mcq_general


class EducationalQAMCQGeneralGenerator:
    def __init__(self, llm, prompt_file: str = None, batch_size: int = 1, diversity_context: str = ""):
        self.llm = llm
        self.batch_size = batch_size
        self.diversity_context = diversity_context

    def generate_batch(self):
        prompt = build_batched_prompt(
            schema=EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA,
            task_instruction=EDUCATIONAL_QA_MCQ_GENERAL_TASK,
            batch_size=self.batch_size,
            diversity_context=self.diversity_context,
        )
        objs = self.llm.generate_batch(prompt, self.batch_size)
        return [repair_educational_qa_mcq_general(obj) for obj in objs]
