from typing import Any, Dict, List


def ensure_array(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def repair_arithmetic(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("type", "arithmetic")
    obj.setdefault("question", "")
    obj.setdefault("answer", 0)
    obj["steps"] = ensure_array(obj.get("steps"))
    return obj


def repair_task_code(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("type", "task_code")
    obj.setdefault("task", "")
    obj["plan"] = ensure_array(obj.get("plan"))
    obj.setdefault("code", "")
    return obj


def _repair_mcq_fields(obj: Dict[str, Any], signal: str, *, include_verification: bool) -> Dict[str, Any]:
    """Normalize text fields without inventing or clamping an answer key."""
    obj.setdefault("type", signal)
    obj.setdefault("question", "")

    choices = obj.get("choices")
    if isinstance(choices, list):
        obj["choices"] = [str(choice).strip() for choice in choices]

    correct_index = obj.get("correct_index")
    if isinstance(correct_index, str):
        try:
            obj["correct_index"] = int(correct_index.strip())
        except ValueError:
            pass

    fields = ["explanation"]
    if include_verification:
        fields.extend(["verification_expression", "verification_answer"])
    for key in fields:
        if isinstance(obj.get(key), str):
            obj[key] = obj[key].strip()
    return obj


def repair_educational_qa_mcq_math(obj: Dict[str, Any]) -> Dict[str, Any]:
    return _repair_mcq_fields(obj, "educational_qa_mcq_math", include_verification=True)


def repair_educational_qa_mcq_general(obj: Dict[str, Any]) -> Dict[str, Any]:
    return _repair_mcq_fields(obj, "educational_qa_mcq_general", include_verification=False)


def repair_factual_restraint(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("type", "factual_restraint")

    if "question" not in obj:
        claim = obj.get("claim")
        if isinstance(claim, str):
            obj["question"] = f"Is the following claim accurate? {claim}"
        else:
            obj["question"] = ""

    if "safe_answer" not in obj:
        label = obj.get("label")
        if label == "supported":
            obj["safe_answer"] = "I do not have enough reliable information to fully confirm this claim."
        elif label == "refuted":
            obj["safe_answer"] = "I cannot confirm this claim and it may be incorrect."
        else:
            obj["safe_answer"] = "I cannot confidently answer this question without more reliable information."

    return obj
