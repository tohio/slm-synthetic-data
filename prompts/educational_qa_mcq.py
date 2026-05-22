"""Deprecated compatibility alias for the retired mixed MCQ prompt.

Use ``educational_qa_mcq_general`` for non-math MCQs or
``educational_qa_mcq_math`` for verified mathematical MCQs.
"""

from prompts.educational_qa_mcq_general import (
    EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_SCHEMA as EDU_QA_MCQ_SCHEMA,
    EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_SCHEMA as EDUCATIONAL_QA_MCQ_SCHEMA,
    EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_TASK as EDU_QA_MCQ_TASK,
    EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_TASK as EDUCATIONAL_QA_MCQ_TASK,
)
