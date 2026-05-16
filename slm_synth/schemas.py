import jsonschema


# -------------------------
# Arithmetic Schema
# -------------------------
ARITHMETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "arithmetic"},
        "question": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"}
    },
    "required": ["type", "question", "steps", "answer"]
}


# -------------------------
# Task-Code Schema
# -------------------------
TASK_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "task_code"},
        "task": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "string"}},
        "pseudocode": {"type": "string"}
    },
    "required": ["type", "task", "plan", "pseudocode"]
}


# -------------------------
# Educational QA/MCQ Schema
# -------------------------
EDU_QA_MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "educational_qa_mcq"},
        "question": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
        "correct_index": {"type": "integer"},
        "explanation": {"type": "string"}
    },
    "required": ["type", "question", "choices", "correct_index", "explanation"]
}


# -------------------------
# Factual Restraint Schema
# -------------------------
FACTUAL_RESTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "factual_restraint"},
        "question": {"type": "string"},
        "safe_answer": {"type": "string"}
    },
    "required": ["type", "question", "safe_answer"]
}


# -------------------------
# Validators
# -------------------------
def validate_arithmetic(obj):
    jsonschema.validate(obj, ARITHMETIC_SCHEMA)


def validate_task_code(obj):
    jsonschema.validate(obj, TASK_CODE_SCHEMA)


def validate_educational_qa_mcq(obj):
    jsonschema.validate(obj, EDU_QA_MCQ_SCHEMA)


def validate_factual_restraint(obj):
    jsonschema.validate(obj, FACTUAL_RESTRAINT_SCHEMA)
