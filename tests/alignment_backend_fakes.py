"""Reusable plain-text backend fakes for generic alignment generation tests."""

from __future__ import annotations

import json
from threading import Lock


class AcceptingAdjudicatorBackend:
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        if "Return exactly two labeled lines" in prompt:
            text = "AGREE: YES\nREASON: The judge correctly applied the supplied evidence."
        else:
            text = "ASSESSABLE: YES\nDECISION: ACCEPT\nREASON: The candidate is grounded and satisfies the brief."
        return {"text": text, "telemetry": {"usage": {"total_tokens": 5}}}


class StagedDPOBackend:
    """Adapt complete-row DPO test renderers to the two plain-text stages."""

    def __init__(self, delegate):
        self.delegate = delegate
        self._rows = {}
        self._lock = Lock()

    def generate_text_with_metadata(self, *, prompt, system_prompt):
        if "Input specs:\n" in prompt:
            specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
            spec_by_id = {spec["id"]: spec for spec in specs}
            result = self.delegate.generate_text_with_metadata(prompt=prompt, system_prompt=system_prompt)
            payload = json.loads(result["text"])
            rows = payload["items"]
            for row, spec in zip(rows, specs, strict=True):
                row["id"] = spec["id"]
                mode = spec_by_id[row["id"]]["metadata"]["output_mode"]
                row.pop("tools", None)
                final = row["chosen"][-1]
                if mode == "structured_json": final["content"] = '{"result":"A correct and complete response."}'
                elif mode == "table": final["content"] = "| result |\n|---|\n| correct |"
                elif mode == "code": final["content"] = "```python\nreturn True\n```"
            with self._lock:
                self._rows.update({row["id"]: row for row in rows})
            chosen = [{key: row[key] for key in ("prompt", "chosen")} for row in rows]
            return {"text": json.dumps({"items": chosen}), "telemetry": result.get("telemetry", {})}
        if "Input specs and chosen candidates:\n" in prompt:
            payload = json.loads(prompt.split("Input specs and chosen candidates:\n", 1)[1])
            with self._lock:
                items = [{"rejected": self._rows[entry["spec"]["id"]]["rejected"]} for entry in payload["items"]]
            return {"text": json.dumps({"items": items}), "telemetry": {"usage": {"total_tokens": 3}}}
        raise AssertionError("unknown DPO stage")
