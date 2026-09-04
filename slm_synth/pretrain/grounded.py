from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slm_synth.pretrain.artifacts import GroundedArtifact
from slm_synth.pretrain.artifacts.planning import (
    BASE_FACTORY_MAP,
    is_derived_artifact,
)
from slm_synth.pretrain.artifacts.quality import assert_valid_artifacts
from slm_synth.pretrain.record_quality import validate_record
from slm_synth.llm import RetryableProviderExhaustedError, StructuredRenderedResponseError


FACTORY_MAP = BASE_FACTORY_MAP


class GroundedTransientProviderBatchError(RuntimeError):
    """A retryable provider outage exhausted one attempt window; batch must be retried."""

    def __init__(
        self,
        *,
        signal: str,
        batch_id: int,
        artifacts: list[GroundedArtifact],
        telemetry: dict[str, Any],
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.signal = signal
        self.batch_id = int(batch_id)
        self.artifacts = artifacts
        self.telemetry = telemetry or {}


class GroundedRenderedBatchError(ValueError):
    """A completed renderer response whose records cannot safely be persisted."""

    def __init__(
        self,
        *,
        signal: str,
        batch_id: int,
        artifacts: list[GroundedArtifact],
        telemetry: dict[str, Any],
        reason: str,
        returned_artifact_ids: list[Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.signal = signal
        self.batch_id = int(batch_id)
        self.artifacts = artifacts
        self.telemetry = telemetry or {}
        self.returned_artifact_ids = returned_artifact_ids


class GroundedBatchStore:
    """Persist completed model requests atomically and rebuild raw JSONL safely."""

    def __init__(self, output_dir: Path, signal: str):
        self.signal = signal
        self.batch_dir = output_dir / "manifests" / "grounded" / signal / "batches"
        self.failed_batch_dir = output_dir / "manifests" / "grounded" / signal / "failed_batches"
        self.raw_path = output_dir / "raw" / f"{signal}.jsonl"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.failed_batch_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: int) -> Path:
        return self.batch_dir / f"batch_{batch_id:09d}.json"

    def _failed_path(self, batch_id: int) -> Path:
        return self.failed_batch_dir / f"batch_{batch_id:09d}.json"

    def completed_batch_ids(self) -> list[int]:
        return sorted(int(path.stem.split("_")[1]) for path in self.batch_dir.glob("batch_*.json"))

    def failed_batch_ids(self) -> list[int]:
        return sorted(int(path.stem.split("_")[1]) for path in self.failed_batch_dir.glob("batch_*.json"))

    def terminal_batch_ids(self) -> list[int]:
        return sorted(set(self.completed_batch_ids()) | set(self.failed_batch_ids()))

    def terminal_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for batch_id in self.completed_batch_ids():
            payload = self._load(self._path(batch_id))
            ranges.append((batch_id, len(payload.get("records", []))))
        for batch_id in self.failed_batch_ids():
            payload = self._load(self._failed_path(batch_id))
            ranges.append((batch_id, int(payload.get("planned_rows", 0) or 0)))
        return sorted((start, size) for start, size in ranges if size > 0)

    def next_candidate_index(self) -> int:
        """Return the first never-attempted deterministic candidate index."""
        ranges = self.terminal_ranges()
        if not ranges:
            return 0
        return max(start + size for start, size in ranges)

    def write_completed(
        self,
        *,
        batch_id: int,
        artifacts: list[GroundedArtifact],
        records: list[dict[str, Any]],
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "batch_id": int(batch_id),
            "signal": self.signal,
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "records": records,
            "telemetry": telemetry or {},
        }
        final_path = self._path(batch_id)
        temp_path = final_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)

    def write_failed(
        self,
        *,
        batch_id: int,
        planned_rows: int,
        artifacts: list[GroundedArtifact],
        error: Exception,
        telemetry: dict[str, Any] | None = None,
        returned_artifact_ids: list[Any] | None = None,
    ) -> None:
        payload = {
            "batch_id": int(batch_id),
            "signal": self.signal,
            "status": "dropped_transient_rendered_failure",
            "planned_rows": int(planned_rows),
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "error_type": type(error).__name__,
            "error": str(error),
            "returned_artifact_ids": returned_artifact_ids,
            "telemetry": telemetry or {},
        }
        final_path = self._failed_path(batch_id)
        temp_path = final_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)

    def materialize_raw(self) -> int:
        temp_path = self.raw_path.with_suffix(".tmp")
        rows = 0
        with temp_path.open("w", encoding="utf-8") as handle:
            for path in self._completed_paths():
                for record in self._load(path).get("records", []):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.raw_path)
        return rows

    def _completed_paths(self) -> list[Path]:
        return [self._path(batch_id) for batch_id in self.completed_batch_ids()]

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def telemetry_summary(self) -> dict[str, float | int]:
        batches = 0
        dropped_batches = 0
        dropped_rows = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0
        aggregate_request_seconds = 0.0
        retries = 0
        retryable_provider_retries = 0
        retry_sleep_seconds = 0.0
        adaptive_window_increases = 0
        adaptive_window_decreases = 0
        adaptive_admission_wait_seconds = 0.0
        adaptive_peak_in_flight_limit = 0
        adaptive_min_in_flight_limit: int | None = None
        max_adaptive_cooldown_seconds = 0.0
        adaptive_batch_size_observed_minimum: int | None = None
        adaptive_batch_size_observed_peak = 0
        adaptive_batch_size_increases = 0
        adaptive_batch_size_decreases = 0
        adaptive_batch_size_failures = 0

        for path in self._completed_paths():
            telemetry = self._load(path).get("telemetry", {}) or {}
            usage = telemetry.get("usage", {}) or {}
            batches += 1
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            cost += float(usage.get("cost", 0.0) or 0.0)
            aggregate_request_seconds += _request_seconds(telemetry)
            retries += int(telemetry.get("retry_count", 0) or 0)
            retryable_provider_retries += int(telemetry.get("retryable_provider_retries", 0) or 0)
            retry_sleep_seconds += float(telemetry.get("retry_sleep_seconds", 0.0) or 0.0)
            adaptive_window_increases += int(telemetry.get("adaptive_window_increases", 0) or 0)
            adaptive_window_decreases += int(telemetry.get("adaptive_window_decreases", 0) or 0)
            adaptive_admission_wait_seconds += float(telemetry.get("adaptive_admission_wait_seconds", 0.0) or 0.0)
            adaptive_peak_in_flight_limit = max(
                adaptive_peak_in_flight_limit,
                int(telemetry.get("adaptive_peak_in_flight_limit", 0) or 0),
            )
            observed_min = telemetry.get("adaptive_min_in_flight_limit")
            if observed_min is not None:
                observed_min = int(observed_min)
                adaptive_min_in_flight_limit = (
                    observed_min
                    if adaptive_min_in_flight_limit is None
                    else min(adaptive_min_in_flight_limit, observed_min)
                )
            max_adaptive_cooldown_seconds = max(
                max_adaptive_cooldown_seconds,
                float(telemetry.get("max_adaptive_cooldown_seconds", 0.0) or 0.0),
            )
            observed_batch_min = telemetry.get("adaptive_batch_size_observed_minimum")
            if observed_batch_min is not None:
                observed_batch_min = int(observed_batch_min)
                adaptive_batch_size_observed_minimum = (
                    observed_batch_min
                    if adaptive_batch_size_observed_minimum is None
                    else min(adaptive_batch_size_observed_minimum, observed_batch_min)
                )
            adaptive_batch_size_observed_peak = max(
                adaptive_batch_size_observed_peak,
                int(telemetry.get("adaptive_batch_size_observed_peak", 0) or 0),
            )
            adaptive_batch_size_increases = max(
                adaptive_batch_size_increases,
                int(telemetry.get("adaptive_batch_size_increases", 0) or 0),
            )
            adaptive_batch_size_decreases = max(
                adaptive_batch_size_decreases,
                int(telemetry.get("adaptive_batch_size_decreases", 0) or 0),
            )
            adaptive_batch_size_failures = max(
                adaptive_batch_size_failures,
                int(telemetry.get("adaptive_batch_size_failures", 0) or 0),
            )

        for batch_id in self.failed_batch_ids():
            payload = self._load(self._failed_path(batch_id))
            telemetry = payload.get("telemetry", {}) or {}
            usage = telemetry.get("usage", {}) or {}
            dropped_batches += 1
            dropped_rows += int(payload.get("planned_rows", 0) or 0)
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            cost += float(usage.get("cost", 0.0) or 0.0)
            aggregate_request_seconds += _request_seconds(telemetry)
            retries += int(telemetry.get("retry_count", 0) or 0)
            retryable_provider_retries += int(telemetry.get("retryable_provider_retries", 0) or 0)
            retry_sleep_seconds += float(telemetry.get("retry_sleep_seconds", 0.0) or 0.0)
            adaptive_window_increases += int(telemetry.get("adaptive_window_increases", 0) or 0)
            adaptive_window_decreases += int(telemetry.get("adaptive_window_decreases", 0) or 0)
            adaptive_admission_wait_seconds += float(telemetry.get("adaptive_admission_wait_seconds", 0.0) or 0.0)
            adaptive_peak_in_flight_limit = max(
                adaptive_peak_in_flight_limit,
                int(telemetry.get("adaptive_peak_in_flight_limit", 0) or 0),
            )
            observed_min = telemetry.get("adaptive_min_in_flight_limit")
            if observed_min is not None:
                observed_min = int(observed_min)
                adaptive_min_in_flight_limit = (
                    observed_min
                    if adaptive_min_in_flight_limit is None
                    else min(adaptive_min_in_flight_limit, observed_min)
                )
            max_adaptive_cooldown_seconds = max(
                max_adaptive_cooldown_seconds,
                float(telemetry.get("max_adaptive_cooldown_seconds", 0.0) or 0.0),
            )
            observed_batch_min = telemetry.get("adaptive_batch_size_observed_minimum")
            if observed_batch_min is not None:
                observed_batch_min = int(observed_batch_min)
                adaptive_batch_size_observed_minimum = (
                    observed_batch_min
                    if adaptive_batch_size_observed_minimum is None
                    else min(adaptive_batch_size_observed_minimum, observed_batch_min)
                )
            adaptive_batch_size_observed_peak = max(
                adaptive_batch_size_observed_peak,
                int(telemetry.get("adaptive_batch_size_observed_peak", 0) or 0),
            )
            adaptive_batch_size_increases = max(
                adaptive_batch_size_increases,
                int(telemetry.get("adaptive_batch_size_increases", 0) or 0),
            )
            adaptive_batch_size_decreases = max(
                adaptive_batch_size_decreases,
                int(telemetry.get("adaptive_batch_size_decreases", 0) or 0),
            )
            adaptive_batch_size_failures = max(
                adaptive_batch_size_failures,
                int(telemetry.get("adaptive_batch_size_failures", 0) or 0),
            )

        return {
            "batches": batches,
            "dropped_batches": dropped_batches,
            "dropped_rows": dropped_rows,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "aggregate_request_seconds": round(aggregate_request_seconds, 3),
            "retry_count": retries,
            "retryable_provider_retries": retryable_provider_retries,
            "retry_sleep_seconds": round(retry_sleep_seconds, 3),
            "adaptive_window_increases": adaptive_window_increases,
            "adaptive_window_decreases": adaptive_window_decreases,
            "adaptive_admission_wait_seconds": round(adaptive_admission_wait_seconds, 3),
            "adaptive_peak_in_flight_limit": adaptive_peak_in_flight_limit,
            "adaptive_min_in_flight_limit": adaptive_min_in_flight_limit or 0,
            "max_adaptive_cooldown_seconds": round(max_adaptive_cooldown_seconds, 3),
            "adaptive_batch_size_observed_minimum": adaptive_batch_size_observed_minimum or 0,
            "adaptive_batch_size_observed_peak": adaptive_batch_size_observed_peak,
            "adaptive_batch_size_increases": adaptive_batch_size_increases,
            "adaptive_batch_size_decreases": adaptive_batch_size_decreases,
            "adaptive_batch_size_failures": adaptive_batch_size_failures,
        }


def _request_seconds(telemetry: dict[str, Any]) -> float:
    if "aggregate_request_seconds" in telemetry:
        return float(telemetry.get("aggregate_request_seconds", 0.0) or 0.0)
    return float(telemetry.get("elapsed_seconds", 0.0) or 0.0)


class GroundedSignalGenerator:
    """Render one homogeneous batch of deterministic grounded artifacts."""

    def __init__(self, signal: str, llm: Any, *, batch_size: int = 32, factory: Any | None = None):
        if signal not in FACTORY_MAP:
            raise ValueError(f"Unsupported grounded signal: {signal}")
        self.signal = signal
        self.llm = llm
        self.batch_size = int(batch_size)
        self.factory = factory or FACTORY_MAP[signal]()

    def response_schema(self, batch_size: int | None = None) -> dict[str, Any]:
        batch_size = int(batch_size or self.batch_size)
        common = {"artifact_id": {"type": "string"}}

        if self.signal == "arithmetic":
            fields = {
                **common,
                "question": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
                "answer": {"type": "string"},
                "verification_expression": {"type": "string"},
            }
            required = ["artifact_id", "question", "steps"]
        elif self.signal == "task_code":
            fields = {
                **common,
                "plan": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                "task": {"type": "string"},
                "code": {"type": "string"},
            }
            required = ["artifact_id", "plan"]
        elif self.signal == "educational_qa_mcq_math":
            fields = {
                **common,
                "explanation": {"type": "string"},
                "question": {"type": "string"},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "verification_expression": {"type": "string"},
                "verification_answer": {"type": "string"},
            }
            required = ["artifact_id", "explanation"]
        elif self.signal == "educational_qa_mcq_general":
            fields = {
                **common,
                "explanation": {"type": "string"},
                "evidence": {"type": "string"},
                "question": {"type": "string"},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
            }
            required = ["artifact_id", "explanation"]
        else:
            fields = {
                **common,
                "safe_answer": {"type": "string"},
                "question": {"type": "string"},
            }
            required = ["artifact_id", "safe_answer"]

        item = {
            "type": "object",
            "properties": fields,
            "required": required,
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": item,
                    "minItems": batch_size,
                    "maxItems": batch_size,
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        }

    def build_prompt(self, artifacts: list[GroundedArtifact]) -> str:
        rows = [
            {"artifact_id": item.artifact_id, "family": item.family, "payload": item.payload}
            for item in artifacts
        ]
        common = (
            "Generate one final synthetic PRETRAINING record component for each grounded artifact below. "
            "Preserve artifact_id exactly and return records in the same order. "
            "Artifacts without generation_mode=derived are authoritative local records: preserve their "
            "facts and emit only the fields requested for the existing finite-artifact path. "
            "For generation_mode=derived, the retained artifact is a capability anchor rather than final "
            "public content. Create a materially new, self-contained record that exercises the same family "
            "and follows every derivation_profile lens. Do not copy the anchor's names, wording, numeric "
            "facts, evidence, task, or code. Return every field explicitly required for derived records. "
            "Return only the JSON object required by the schema.\n\n"
        )
        instructions = {
            "arithmetic": (
                "Finite artifacts: generate a natural learner-facing question and compact worked steps. "
                "Use every required_numeric_literal in the question, introduce no extra numeric quantities, "
                "and do not reveal the held answer in the question. When required_text_literals are present, "
                "preserve each phrase exactly and use the supplied domain, item, source, facts, and reasoning "
                "family materially. The verified answer remains local. "
                "Derived artifacts: generate question, steps, answer, and verification_expression. The "
                "verification expression may use only integer literals, parentheses, +, -, *, and /. It must "
                "evaluate exactly to the integer answer. Every numeric literal in the expression must appear "
                "in the question with the same multiplicity, and the question must introduce no other numeric "
                "quantity. Do not reveal the answer unless it is itself a required operand."
            ),
            "task_code": (
                "Finite artifacts: generate only a faithful 2-to-4 step implementation plan. The public task "
                "and code remain authoritative and local. The plan must cover them, must not mention the held "
                "function name, and must not add behavior absent from the artifact. "
                "Derived artifacts: generate task, plan, and code. The task must begin exactly with "
                "'Write a Python function that', the plan must contain 2 to 4 useful steps without naming the "
                "function, and code must be plain source text containing exactly one complete top-level Python "
                "function. Do not use Markdown fences. The task, plan, and code must agree."
            ),
            "educational_qa_mcq_math": (
                "Finite artifacts: generate only a concise explanation of why the verified answer follows "
                "from the authoritative local question and expression. State the numeric answer exactly and "
                "show the relevant calculation. Question, choices, and verified answer remain local. "
                "Derived artifacts: generate question, four unique integer choices, explanation, "
                "verification_expression, and verification_answer. The expression may use only integer "
                "literals, parentheses, +, -, *, and / and must evaluate exactly to the integer verification "
                "answer. Every expression literal must appear in the question with the same multiplicity, the "
                "question must introduce no other numeric quantity, exactly one choice must equal the answer, "
                "and the explanation must state and justify that answer."
            ),
            "educational_qa_mcq_general": (
                "Finite artifacts: generate only a concise learner-facing explanation showing why the correct "
                "choice follows from the supplied evidence. Evidence, question, choices, and answer remain "
                "local and must not be changed. Do not mention an answer key, held answer, supplied answer, "
                "artifact, prompt, or generation process. "
                "Derived artifacts: generate fresh evidence, question, four unique choices, correct_index, and "
                "explanation. The evidence must contain all facts needed to support exactly the indexed choice, "
                "and the explanation must be faithful to that evidence without referring to generation."
            ),
            "factual_restraint": (
                "Finite artifacts: generate a concise natural user-facing assistant answer to the supplied "
                "question. Follow the behavior requirement without inventing facts or writing policy "
                "instructions such as 'Do not provide'. Use direct language such as 'I can't provide' or "
                "'I can't verify'. Derived artifacts: generate a fresh question and safe_answer that exercise "
                "the same restraint family and behavior in the requested derivation profile. Use different "
                "people, organizations, places, dates, and concrete details from the anchor, and do not invent "
                "an answer to information that is unavailable or inappropriate to provide."
            ),
        }[self.signal]

        return common + instructions + "\n\nGROUNDED ARTIFACTS:\n" + json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _numeric_literals(text: str) -> list[str]:
        return re.findall(r"(?<![\w])-?\d+(?![\w])", text)

    def _require_expression_literals_in_question(
        self,
        *,
        artifact: GroundedArtifact,
        question: str,
        expression: str,
    ) -> None:
        observed = Counter(self._numeric_literals(question))
        required = Counter(self._numeric_literals(expression))
        if not required or observed != required:
            raise ValueError(
                f"Rendered {self.signal} question changed verification facts for {artifact.artifact_id}"
            )

    def _finalize(self, artifact: GroundedArtifact, row: dict[str, Any]) -> dict[str, Any]:
        payload = artifact.payload
        derived = is_derived_artifact(artifact)

        if self.signal == "arithmetic":
            question = str(row.get("question", "")).strip()
            if derived:
                answer = str(row.get("answer", "")).strip()
                expression = str(row.get("verification_expression", "")).strip()
                self._require_expression_literals_in_question(
                    artifact=artifact,
                    question=question,
                    expression=expression,
                )
                required = self._numeric_literals(expression)
            else:
                answer = str(payload["answer"])
                expression = str(payload["expression"])
                observed = self._numeric_literals(question)
                required = list(payload["required_numeric_literals"])
                if Counter(observed) != Counter(required):
                    raise ValueError(
                        f"Rendered arithmetic question changed numeric facts for {artifact.artifact_id}"
                    )
                missing_text = [
                    literal
                    for literal in payload.get("required_text_literals", [])
                    if str(literal).casefold() not in question.casefold()
                ]
                if missing_text:
                    raise ValueError(
                        f"Rendered arithmetic question dropped semantic context for {artifact.artifact_id}: "
                        + ", ".join(str(value) for value in missing_text)
                    )

            if answer in self._numeric_literals(question) and answer not in required:
                raise ValueError(f"Rendered arithmetic question leaks answer for {artifact.artifact_id}")

            record = {
                "type": "arithmetic",
                "question": question,
                "steps": row.get("steps"),
                "answer": answer,
                "verification_expression": expression,
                "verification_answer": answer,
            }
            result = validate_record(
                "arithmetic", record, require_arithmetic_verification=True
            )

        elif self.signal == "task_code":
            task = str(row.get("task") if derived else payload.get("task", "")).strip()
            code = str(row.get("code") if derived else payload.get("code", "")).strip()
            lower = task.lower()
            plan = row.get("plan")

            if not lower.startswith("write a python function that") or "```" in task or "\ndef " in lower:
                raise ValueError(
                    f"Grounded task_code task is not a clean instruction for {artifact.artifact_id}"
                )
            tree = ast.parse(code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError(
                    f"Grounded task_code code is not one top-level function for {artifact.artifact_id}"
                )
            function_name = tree.body[0].name
            if isinstance(plan, list) and function_name.casefold() in " ".join(map(str, plan)).casefold():
                raise ValueError(
                    f"Rendered task_code plan leaks the held function name for {artifact.artifact_id}"
                )

            record = {"type": "task_code", "task": task, "plan": plan, "code": code}
            result = validate_record("task_code", record)

        elif self.signal == "educational_qa_mcq_math":
            if derived:
                question = str(row.get("question", "")).strip()
                choices = row.get("choices")
                expression = str(row.get("verification_expression", "")).strip()
                answer = str(row.get("verification_answer", "")).strip()
                self._require_expression_literals_in_question(
                    artifact=artifact,
                    question=question,
                    expression=expression,
                )
                required = self._numeric_literals(expression)
                if answer in self._numeric_literals(question) and answer not in required:
                    raise ValueError(
                        f"Rendered math MCQ question leaks answer for {artifact.artifact_id}"
                    )
                correct_index = 0  # Validation derives the authoritative index from the verified answer.
            else:
                question = payload["question"]
                choices = payload["choices"]
                correct_index = payload["correct_index"]
                expression = payload["expression"]
                answer = payload["answer"]

            record = {
                "type": "educational_qa_mcq_math",
                "question": question,
                "choices": choices,
                "correct_index": correct_index,
                "explanation": row.get("explanation"),
                "verification_expression": expression,
                "verification_answer": answer,
            }
            result = validate_record(
                "educational_qa_mcq_math",
                record,
                require_mcq_verification=True,
            )

        elif self.signal == "educational_qa_mcq_general":
            explanation = row.get("explanation")
            if derived:
                evidence = row.get("evidence")
                question = row.get("question")
                choices = row.get("choices")
                correct_index = row.get("correct_index")
            else:
                evidence = payload["evidence"]
                question = payload["question"]
                choices = payload["choices"]
                correct_index = payload["correct_index"]
                count_match = re.search(
                    r"values\s*=\s*(\[[^\]]+\])\s*[\r\n]+"
                    r"result\s*=\s*values\.count\(([-]?\d+)\)",
                    str(evidence),
                    re.I,
                )
                if count_match:
                    values = ast.literal_eval(count_match.group(1))
                    target = int(count_match.group(2))
                    expected = values.count(target)
                    explanation = (
                        f"The list `values` contains {expected} occurrence"
                        f"{'' if expected == 1 else 's'} of the integer {target}, "
                        f"so `values.count({target})` returns {expected}."
                    )

            record = {
                "type": "educational_qa_mcq_general",
                "evidence": evidence,
                "question": question,
                "choices": choices,
                "correct_index": correct_index,
                "explanation": explanation,
            }
            result = validate_record("educational_qa_mcq_general", record)

        else:
            record = {
                "type": "factual_restraint",
                "question": row.get("question") if derived else payload["question"],
                "safe_answer": row.get("safe_answer"),
            }
            result = validate_record("factual_restraint", record)

        if not result.ok:
            raise ValueError(
                f"Rendered {self.signal} record failed validation for {artifact.artifact_id}: {result.issues}"
            )

        return record

    def generate_batch(self, batch_id: int) -> tuple[list[GroundedArtifact], list[dict[str, Any]], dict[str, Any]]:
        return self.generate_range(batch_id * self.batch_size, self.batch_size, batch_id=batch_id)

    def generate_range(
        self,
        start_index: int,
        batch_size: int,
        *,
        batch_id: int | None = None,
    ) -> tuple[list[GroundedArtifact], list[dict[str, Any]], dict[str, Any]]:
        batch_id = int(start_index if batch_id is None else batch_id)
        batch_size = int(batch_size)
        artifacts = [self.factory.build(index) for index in range(int(start_index), int(start_index) + batch_size)]
        assert_valid_artifacts(artifacts)
        prompt = self.build_prompt(artifacts)

        if hasattr(self.llm, "generate_structured_object_with_metadata"):
            try:
                result = self.llm.generate_structured_object_with_metadata(
                    prompt=prompt,
                    schema=self.response_schema(batch_size),
                    schema_name=f"grounded_{self.signal}_batch_{batch_size}",
                )
            except RetryableProviderExhaustedError as exc:
                raise GroundedTransientProviderBatchError(
                    signal=self.signal,
                    batch_id=batch_id,
                    artifacts=artifacts,
                    telemetry=exc.telemetry,
                    reason=str(exc),
                ) from exc
            except StructuredRenderedResponseError as exc:
                raise GroundedRenderedBatchError(
                    signal=self.signal,
                    batch_id=batch_id,
                    artifacts=artifacts,
                    telemetry=exc.telemetry,
                    reason=str(exc),
                ) from exc

            response = result["data"]
            telemetry = result.get("telemetry", {})
        else:
            response = self.llm.generate_structured_object(
                prompt=prompt,
                schema=self.response_schema(batch_size),
                schema_name=f"grounded_{self.signal}_batch_{batch_size}",
            )
            telemetry = {}

        returned_ids: list[Any] | None = None

        try:
            rows = response.get("records") if isinstance(response, dict) else None
            if not isinstance(rows, list) or len(rows) != batch_size:
                raise ValueError(f"Expected {batch_size} grounded {self.signal} records")

            expected = {artifact.artifact_id: artifact for artifact in artifacts}
            returned_ids = [row.get("artifact_id") for row in rows if isinstance(row, dict)]

            if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(expected):
                raise ValueError(f"Grounded {self.signal} response has missing, duplicate, or unexpected artifact IDs")

            records = [self._finalize(expected[row["artifact_id"]], row) for row in rows]

        except ValueError as exc:
            raise GroundedRenderedBatchError(
                signal=self.signal,
                batch_id=batch_id,
                artifacts=artifacts,
                telemetry=telemetry,
                reason=str(exc),
                returned_artifact_ids=returned_ids,
            ) from exc

        return artifacts, records, telemetry
