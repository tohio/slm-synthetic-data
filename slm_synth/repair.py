# slm_synth/repair.py

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


def repair_educational_qa_mcq(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("type", "educational_qa_mcq")
    obj.setdefault("question", "")
    choices = obj.get("choices") or []
    if not isinstance(choices, list):
        choices = [str(choices)]
    choices = [str(c) for c in choices]
    while len(choices) < 4:
        choices.append("")
    if len(choices) > 4:
        choices = choices[:4]
    obj["choices"] = choices

    if "correct_index" not in obj:
        answer = obj.get("answer")
        if isinstance(answer, str) and answer in choices:
            obj["correct_index"] = choices.index(answer)
        else:
            obj["correct_index"] = 0

    if not isinstance(obj.get("correct_index"), int):
        obj["correct_index"] = 0

    obj.setdefault("explanation", "")
    return obj


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
