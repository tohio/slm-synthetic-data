import pytest
from slm_synth.schemas import (
    validate_arithmetic,
    validate_task_code,
    validate_educational_qa_mcq,
    validate_factual_restraint,
)

def test_arithmetic_schema_valid():
    obj = {
        "type": "arithmetic",
        "question": "What is 2+2?",
        "steps": ["2+2=4"],
        "answer": "4"
    }
    validate_arithmetic(obj)

def test_task_code_schema_valid():
    obj = {
        "type": "task_code",
        "task": "Sort a list",
        "plan": ["Read list", "Sort list", "Return result"],
        "pseudocode": "list.sort()"
    }
    validate_task_code(obj)

def test_educational_qa_mcq_schema_valid():
    obj = {
        "type": "educational_qa_mcq",
        "question": "What planet is third from the Sun?",
        "choices": ["Mars", "Earth", "Venus", "Jupiter"],
        "correct_index": 1,
        "explanation": "Earth is third."
    }
    validate_educational_qa_mcq(obj)

def test_factual_restraint_schema_valid():
    obj = {
        "type": "factual_restraint",
        "question": "Who will win the next World Cup?",
        "safe_answer": "I cannot predict future events."
    }
    validate_factual_restraint(obj)
