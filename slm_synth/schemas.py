import jsonschema


ARITHMETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "arithmetic"},
        "question": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
    },
    "required": ["type", "question", "steps", "answer"],
}

TASK_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "task_code"},
        "task": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "string"}},
        "code": {"type": "string"},
    },
    "required": ["type", "task", "plan", "code"],
}

# Raw math-MCQ records include temporary metadata used during validation.
EDUCATIONAL_QA_MCQ_MATH_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "educational_qa_mcq_math"},
        "question": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
        "explanation": {"type": "string"},
        "verification_expression": {"type": "string"},
        "verification_answer": {"type": "string"},
    },
    "required": [
        "type",
        "question",
        "choices",
        "correct_index",
        "explanation",
        "verification_expression",
        "verification_answer",
    ],
}

EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "educational_qa_mcq_general"},
        "evidence": {"type": "string"},
        "question": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
        "explanation": {"type": "string"},
    },
    "required": ["type", "evidence", "question", "choices", "correct_index", "explanation"],
}

# Compatibility alias: the old mixed MCQ path is retired in the generated mix.
EDUCATIONAL_QA_MCQ_SCHEMA = EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA

FACTUAL_RESTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "factual_restraint"},
        "question": {"type": "string"},
        "safe_answer": {"type": "string"},
    },
    "required": ["type", "question", "safe_answer"],
}


def validate_arithmetic(obj):
    jsonschema.validate(obj, ARITHMETIC_SCHEMA)


def validate_task_code(obj):
    jsonschema.validate(obj, TASK_CODE_SCHEMA)


def validate_educational_qa_mcq_math(obj):
    jsonschema.validate(obj, EDUCATIONAL_QA_MCQ_MATH_SCHEMA)


def validate_educational_qa_mcq_general(obj):
    jsonschema.validate(obj, EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA)


def validate_educational_qa_mcq(obj):
    """Backward-compatible validator for the general, non-math MCQ schema."""
    validate_educational_qa_mcq_general(obj)


def validate_factual_restraint(obj):
    jsonschema.validate(obj, FACTUAL_RESTRAINT_SCHEMA)
