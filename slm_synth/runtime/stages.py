from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

from slm_synth.llm import StructuredRenderedResponseError


class Progress:
    """Progress reporting shape proven by the finalized one-off pipelines."""

    def __init__(
        self,
        stage: str,
        total_batches: int,
        total_items: int,
        batch_size: int,
        concurrency: int,
    ):
        self.stage = stage
        self.total_batches = total_batches
        self.total_items = total_items
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.completed_batches = 0
        self.completed_items = 0
        self.failures = 0
        self.started = time.monotonic()
        self.lock = threading.Lock()

        print(
            f"[{stage}] start items={total_items} batches={total_batches} "
            f"batch_size={batch_size} concurrency={concurrency}",
            flush=True,
        )

    def done(self, item_count: int) -> None:
        with self.lock:
            self.completed_batches += 1
            self.completed_items += item_count
            elapsed = time.monotonic() - self.started
            print(
                f"[{self.stage}] progress "
                f"batches={self.completed_batches}/{self.total_batches} "
                f"items={self.completed_items}/{self.total_items} "
                f"failures={self.failures} elapsed={elapsed:.1f}s",
                flush=True,
            )

    def failed(self, message: str) -> None:
        with self.lock:
            self.failures += 1
            elapsed = time.monotonic() - self.started
            print(
                f"[{self.stage}] failure count={self.failures} "
                f"elapsed={elapsed:.1f}s error={message!r}",
                flush=True,
            )

    def split(
        self,
        *,
        original_items: int,
        left_items: int,
        right_items: int,
    ) -> None:
        with self.lock:
            self.total_batches += 1
            elapsed = time.monotonic() - self.started
            print(
                f"[{self.stage}] isolate split_items={original_items} "
                f"into={left_items}+{right_items} "
                f"batches={self.completed_batches}/{self.total_batches} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    def finish(self) -> None:
        elapsed = time.monotonic() - self.started
        print(
            f"[{self.stage}] complete "
            f"items={self.completed_items}/{self.total_items} "
            f"batches={self.completed_batches}/{self.total_batches} "
            f"failures={self.failures} elapsed={elapsed:.1f}s",
            flush=True,
        )


def _json_safe_batch(batch: Any) -> Any:
    if hasattr(batch, "__dataclass_fields__"):
        return asdict(batch)
    if isinstance(batch, (dict, list, tuple, str, int, float, bool)) or batch is None:
        return batch
    return repr(batch)


def run_model_stage_with_isolation(
    *,
    stage: str,
    batches: Sequence[Any],
    item_count: Callable[[Any], int],
    worker: Callable[[Any], Any],
    split_batch: Callable[[Any], tuple[Any, Any] | None],
    concurrency: int,
    batch_size_display: int,
    max_attempts: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run a model stage with bounded retry and recursive fault isolation.

    Successful work is retained. A failed multi-item batch is retried up to
    ``max_attempts`` and then recursively split. A one-item batch that still
    fails is recorded and excluded so the rest of the pipeline can continue.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    progress = Progress(
        stage=stage,
        total_batches=len(batches),
        total_items=sum(item_count(batch) for batch in batches),
        batch_size=batch_size_display,
        concurrency=concurrency,
    )

    results: list[Any] = []
    failures: list[dict[str, Any]] = []
    pending: list[tuple[int, Any]] = [(1, batch) for batch in batches]

    while pending:
        next_pending: list[tuple[int, Any]] = []

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(worker, batch): (attempt, batch)
                for attempt, batch in pending
            }

            for future in as_completed(futures):
                attempt, batch = futures[future]
                count = item_count(batch)

                try:
                    result = future.result()
                except Exception as exc:
                    progress.failed(
                        f"attempt={attempt}/{max_attempts} "
                        f"items={count} error={exc}"
                    )

                    split = split_batch(batch)

                    # A rendered structured response already consumed a successful
                    # provider call. Repeating the same multi-item payload tends to
                    # reproduce the same serialization/truncation failure, so isolate
                    # it immediately. Provider/transport failures keep normal retries.
                    if isinstance(exc, StructuredRenderedResponseError) and split is not None:
                        left, right = split
                        left_count = item_count(left)
                        right_count = item_count(right)

                        if left_count < 1 or right_count < 1:
                            raise RuntimeError(
                                f"{stage} split produced an empty child batch"
                            )
                        if left_count + right_count != count:
                            raise RuntimeError(
                                f"{stage} split changed item cardinality: "
                                f"{count} -> {left_count}+{right_count}"
                            )

                        progress.split(
                            original_items=count,
                            left_items=left_count,
                            right_items=right_count,
                        )
                        next_pending.append((1, left))
                        next_pending.append((1, right))
                        continue

                    if attempt < max_attempts:
                        print(
                            f"[{stage}] retry batch_items={count} "
                            f"next_attempt={attempt + 1}/{max_attempts}",
                            flush=True,
                        )
                        next_pending.append((attempt + 1, batch))
                        continue

                    if split is not None:
                        left, right = split
                        left_count = item_count(left)
                        right_count = item_count(right)

                        if left_count < 1 or right_count < 1:
                            raise RuntimeError(
                                f"{stage} split produced an empty child batch"
                            )
                        if left_count + right_count != count:
                            raise RuntimeError(
                                f"{stage} split changed item cardinality: "
                                f"{count} -> {left_count}+{right_count}"
                            )

                        progress.split(
                            original_items=count,
                            left_items=left_count,
                            right_items=right_count,
                        )
                        next_pending.append((1, left))
                        next_pending.append((1, right))
                        continue

                    error_text = f"{type(exc).__name__}: {exc}"
                    failures.append(
                        {
                            "stage": stage,
                            "items": count,
                            "error": error_text,
                            "batch": _json_safe_batch(batch),
                        }
                    )
                    print(
                        f"[{stage}] isolated_failure items={count} "
                        f"error={error_text!r} action=continue",
                        flush=True,
                    )
                    progress.done(count)
                    continue

                results.append(result)
                progress.done(count)

        pending = next_pending

    progress.finish()
    return results, failures
