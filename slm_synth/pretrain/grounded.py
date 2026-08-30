from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slm_synth.pretrain.artifacts import (
    ArithmeticArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    FactualRestraintArtifactFactory,
    GroundedArtifact,
    TaskCodeArtifactFactory,
)
from slm_synth.pretrain.artifacts.scalable import ScalableArtifactFactory
from slm_synth.pretrain.artifacts.quality import assert_valid_artifacts
from slm_synth.pretrain.record_quality import validate_record
from slm_synth.llm import RetryableProviderExhaustedError, StructuredRenderedResponseError


FACTORY_MAP = {
    "arithmetic": lambda: ScalableArtifactFactory(ArithmeticArtifactFactory()),
    "task_code": lambda: ScalableArtifactFactory(TaskCodeArtifactFactory()),
    "educational_qa_mcq_math": lambda: ScalableArtifactFactory(EducationalQAMCQMathArtifactFactory()),
    "educational_qa_mcq_general": lambda: ScalableArtifactFactory(EducationalQAMCQGeneralArtifactFactory()),
    "factual_restraint": lambda: ScalableArtifactFactory(FactualRestraintArtifactFactory()),
}


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
                "task": {"type": "string"},
                "plan": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                "code": {"type": "string"},
            }
            required = ["artifact_id", "plan"]
        elif self.signal == "educational_qa_mcq_math":
            fields = {
                **common,
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                "explanation": {"type": "string"},
                "verification_expression": {"type": "string"},
                "verification_answer": {"type": "string"},
            }
            required = ["artifact_id", "explanation"]
        elif self.signal == "educational_qa_mcq_general":
            fields = {
                **common,
                "evidence": {"type": "string"},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                "explanation": {"type": "string"},
            }
            required = ["artifact_id", "explanation"]
        else:
            fields = {
                **common,
                "question": {"type": "string"},
                "safe_answer": {"type": "string"},
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
            {
                "artifact_id": item.artifact_id,
                "family": item.family,
                "mode": "derived" if "_derivation" in item.payload else "grounded",
                "payload": item.payload,
            }
            for item in artifacts
        ]
        common = (
            "Generate one final synthetic PRETRAINING record component for each grounded artifact below. "
            "For mode=grounded, every grounded artifact is authoritative. For mode=derived, the seed payload is an "
            "archetype only: follow _derivation and create a materially distinct new case in the same family rather "
            "than copying, renaming, renumbering, or lightly paraphrasing the seed. Preserve artifact_id exactly and "
            "return records in the same order. Return only the JSON object required by the schema.\n\n"
        )
        instructions = {
            "arithmetic": (
                "For each artifact, generate a natural learner-facing question and compact worked steps. "
                "Use every required_numeric_literal in the question, introduce no extra numeric quantities, "
                "and do not reveal the held answer in the question. When required_text_literals are present, "
                "preserve each phrase exactly and use the supplied domain, item, source, facts, and reasoning "
                "family materially; do not reduce contextual artifacts to generic item-count templates. "
                "The verified answer remains local. For mode=derived, instead create a new self-contained integer "
                "problem matching the family and derivation profile; return question, steps, answer, and a Python-style "
                "verification_expression whose integer value exactly equals answer. Use no numeric quantities in the "
                "question beyond those required by that expression."
            ),
            "task_code": (
                "For each valid Python code artifact, generate only a faithful 2-to-4 step implementation plan. "
                "The public task is authoritative and remains local. The plan must cover the supplied task and code, "
                "must not mention the held function name, and must not add behavior absent from the artifact. For "
                "mode=derived, create a materially different Python task and complete implementation in the same broad "
                "algorithmic family, plus a faithful 2-to-4 step plan. The new task/code must not be a variable rename, "
                "constant tweak, wrapper, or cosmetic rewrite of the seed."
            ),
            "educational_qa_mcq_math": (
                "For each artifact, generate only a concise explanation of why the verified answer follows from "
                "the authoritative local question and expression. State the numeric answer exactly, show the "
                "relevant calculation, and do not discuss question generation or the supplied choices. "
                "Question, choices, and verified answer remain local and must not be changed. For mode=derived, create "
                "a new self-contained math MCQ in the same relationship family with four distinct choices, correct_index, "
                "a concise explanation, and a Python-style verification_expression plus verification_answer that agree."
            ),
            "educational_qa_mcq_general": (
                "For each artifact, generate only a concise learner-facing explanation showing why the correct "
                "choice follows from the supplied evidence. Evidence, question, choices, and answer remain local "
                "and must not be changed. Do not mention an answer key, held answer, supplied answer, artifact, "
                "prompt, or generation process. For mode=derived, create a materially different self-contained evidence "
                "case in the same reasoning family, with evidence, question, four distinct choices, correct_index, and "
                "a concise explanation. The answer must be directly supported by the new evidence."
            ),
            "factual_restraint": (
                "For each artifact, generate a concise natural user-facing assistant answer to the supplied question. "
                "Follow the behavior requirement without inventing facts or writing policy instructions such as "
                "'Do not provide'. Use direct language such as 'I can\\'t provide' or 'I can\\'t verify'."
            ),
        }[self.signal]

        return common + instructions + "\n\nGROUNDED ARTIFACTS:\n" + json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _numeric_literals(text: str) -> list[str]:
        return re.findall(r"(?<![\w])-?\d+(?![\w])", text)

    def _finalize(self, artifact: GroundedArtifact, row: dict[str, Any]) -> dict[str, Any]:
        payload = artifact.payload
        derived = "_derivation" in payload

        if derived:
            if self.signal == "arithmetic":
                question = str(row.get("question", "")).strip()
                expression = str(row.get("verification_expression", "")).strip()
                answer = str(row.get("answer", "")).strip()
                record = {
                    "type": "arithmetic",
                    "question": question,
                    "steps": row.get("steps"),
                    "answer": answer,
                    "verification_expression": expression,
                    "verification_answer": answer,
                }
                result = validate_record("arithmetic", record, require_arithmetic_verification=True)
                expression_numbers = self._numeric_literals(expression)
                question_numbers = self._numeric_literals(question)
                if Counter(question_numbers) != Counter(expression_numbers):
                    raise ValueError(
                        f"Derived arithmetic question/expression numeric facts differ for {artifact.artifact_id}"
                    )
            elif self.signal == "task_code":
                task = str(row.get("task", "")).strip()
                code = str(row.get("code", "")).strip()
                if not task.lower().startswith("write a python function that") or "```" in task:
                    raise ValueError(f"Derived task_code task is not a clean instruction for {artifact.artifact_id}")
                record = {"type": "task_code", "task": task, "plan": row.get("plan"), "code": code}
                result = validate_record("task_code", record)
            elif self.signal == "educational_qa_mcq_math":
                record = {
                    "type": "educational_qa_mcq_math",
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                    "correct_index": row.get("correct_index"),
                    "explanation": row.get("explanation"),
                    "verification_expression": row.get("verification_expression"),
                    "verification_answer": row.get("verification_answer"),
                }
                result = validate_record("educational_qa_mcq_math", record, require_mcq_verification=True)
            elif self.signal == "educational_qa_mcq_general":
                record = {
                    "type": "educational_qa_mcq_general",
                    "evidence": row.get("evidence"),
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                    "correct_index": row.get("correct_index"),
                    "explanation": row.get("explanation"),
                }
                result = validate_record("educational_qa_mcq_general", record)
            else:
                record = {
                    "type": "factual_restraint",
                    "question": row.get("question"),
                    "safe_answer": row.get("safe_answer"),
                }
                result = validate_record("factual_restraint", record)

            if not result.ok:
                raise ValueError(
                    f"Rendered derived {self.signal} record failed validation for {artifact.artifact_id}: {result.issues}"
                )
            return result.record

        if self.signal == "arithmetic":
            question = str(row.get("question", "")).strip()
            observed = self._numeric_literals(question)
            required = list(payload["required_numeric_literals"])

            if Counter(observed) != Counter(required):
                raise ValueError(f"Rendered arithmetic question changed numeric facts for {artifact.artifact_id}")
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
            if payload["answer"] in observed and payload["answer"] not in required:
                raise ValueError(f"Rendered arithmetic question leaks answer for {artifact.artifact_id}")

            record = {
                "type": "arithmetic",
                "question": question,
                "steps": row.get("steps"),
                "answer": payload["answer"],
                "verification_expression": payload["expression"],
                "verification_answer": payload["answer"],
            }
            result = validate_record("arithmetic", record, require_arithmetic_verification=True)

        elif self.signal == "task_code":
            task = str(payload.get("task", "")).strip()
            lower = task.lower()
            plan = row.get("plan")

            if not lower.startswith("write a python function that") or "```" in task or "\ndef " in lower:
                raise ValueError(f"Grounded task_code task is not a clean instruction for {artifact.artifact_id}")
            function_name = ast.parse(str(payload["code"])).body[0].name
            if isinstance(plan, list) and function_name.casefold() in " ".join(map(str, plan)).casefold():
                raise ValueError(f"Rendered task_code plan leaks the held function name for {artifact.artifact_id}")

            record = {"type": "task_code", "task": task, "plan": plan, "code": payload["code"]}
            result = validate_record("task_code", record)

        elif self.signal == "educational_qa_mcq_math":
            record = {
                "type": "educational_qa_mcq_math",
                "question": payload["question"],
                "choices": payload["choices"],
                "correct_index": payload["correct_index"],
                "explanation": row.get("explanation"),
                "verification_expression": payload["expression"],
                "verification_answer": payload["answer"],
            }
            result = validate_record("educational_qa_mcq_math", record, require_mcq_verification=True)

        elif self.signal == "educational_qa_mcq_general":
            explanation = row.get("explanation")
            count_match = re.search(
                r"values\s*=\s*(\[[^\]]+\])\s*[\r\n]+"
                r"result\s*=\s*values\.count\(([-]?\d+)\)",
                str(payload.get("evidence", "")),
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
                "evidence": payload["evidence"],
                "question": payload["question"],
                "choices": payload["choices"],
                "correct_index": payload["correct_index"],
                "explanation": explanation,
            }
            result = validate_record("educational_qa_mcq_general", record)

        else:
            record = {
                "type": "factual_restraint",
                "question": payload["question"],
                "safe_answer": row.get("safe_answer"),
            }
            result = validate_record("factual_restraint", record)

        if not result.ok:
            raise ValueError(f"Rendered {self.signal} record failed validation for {artifact.artifact_id}: {result.issues}")

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
