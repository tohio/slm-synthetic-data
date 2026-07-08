"""Shared OpenRouter-backed generation throughput defaults.

These values intentionally match the grounded pretrain configuration posture:
small smoke concurrency, higher target concurrency, bounded batch size, and
pretrain-style adaptive batch recovery.
"""

from __future__ import annotations

DEFAULT_OPENROUTER_BATCH_SIZE = 32
MIN_OPENROUTER_BATCH_SIZE = 1
MAX_OPENROUTER_BATCH_SIZE = 64

DEFAULT_OPENROUTER_SMOKE_CONCURRENCY = 1
DEFAULT_OPENROUTER_TARGET_CONCURRENCY = 4
MIN_OPENROUTER_CONCURRENCY = 1
MAX_OPENROUTER_CONCURRENCY = 1024

DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT = 8
DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE = 4
DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES = 4
