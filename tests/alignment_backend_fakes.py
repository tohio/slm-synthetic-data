"""Reusable structured-backend fakes for generic alignment generation tests."""

from __future__ import annotations

import json
from threading import Lock


class AcceptingAdjudicatorBackend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        payload = json.loads(prompt.split("Candidates:\n", 1)[1])
        items = []
        for entry in payload["items"]:
            spec = entry["spec"]
            constraints = spec.get("constraints", [])
            if schema_name == "sft_quality_adjudication":
                scores = {
                    "correctness": 4, "grounding": 4, "instruction_adherence": 4,
                    "completeness": 4, "coherence": 4,
                }
                item = {"id": spec["id"], "accepted": True, "scores": scores}
            elif schema_name == "dpo_quality_adjudication":
                scores = {
                    "chosen_quality": 4, "rejected_plausibility": 4, "weakness_match": 4,
                    "preference_separation": 4, "collateral_preservation": 4,
                }
                item = {
                    "id": spec["id"], "accepted": True, "scores": scores,
                    "preference_dimension": spec["metadata"]["preference_dimension"],
                    "failure_mode": spec["metadata"]["failure_mode"],
                    "observed_weakness": spec["metadata"]["failure_mode"],
                }
            else:
                raise AssertionError(schema_name)
            item["constraint_results"] = [
                {"constraint_index": index, "passed": True, "reason": "satisfied"}
                for index, _constraint in enumerate(constraints)
            ]
            item["reasons"] = []
            items.append(item)
        return {"data": {"items": items}, "telemetry": {"usage": {"total_tokens": 5}}}


class StagedDPOBackend:
    """Adapt complete-row DPO test renderers to the staged production contract."""

    def __init__(self, delegate):
        self.delegate = delegate
        self._rows = {}
        self._lock = Lock()

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        if schema_name == "dpo_chosen_batch":
            result = self.delegate.generate_structured_object_with_metadata(
                prompt=prompt, schema=schema, schema_name=schema_name
            )
            rows = result["data"]["items"]
            for row in rows:
                mode = row["metadata"]["output_mode"]
                final = row["chosen"][-1]
                if mode == "structured_json":
                    final["content"] = '{"result":"A correct and complete response."}'
                elif mode == "table":
                    final["content"] = "| result |\n|---|\n| correct |"
                elif mode == "code":
                    final["content"] = "```python\nreturn True\n```"
            with self._lock:
                self._rows.update({row["id"]: row for row in rows})
            chosen_items = []
            for row in rows:
                chosen = {key: row[key] for key in ("id", "prompt", "chosen", "metadata")}
                if "tools" in row:
                    chosen["tools"] = row["tools"]
                chosen_items.append(chosen)
            return {**result, "data": {"items": chosen_items}}
        if schema_name == "dpo_rejected_batch":
            payload = json.loads(prompt.split("Input specs and chosen candidates:\n", 1)[1])
            with self._lock:
                items = [
                    {"id": entry["spec"]["id"], "rejected": self._rows[entry["spec"]["id"]]["rejected"]}
                    for entry in payload["items"]
                ]
            return {"data": {"items": items}, "telemetry": {"usage": {"total_tokens": 3}}}
        raise AssertionError(schema_name)
