import random
import time

class RateLimiter:
    def __init__(self, cfg):
        rl = cfg.get("rate_limit", {})

        self.delay_ms = rl.get("delay_ms", 0)
        self.jitter_ms = rl.get("jitter_ms", 0)
        self.max_concurrency = rl.get("max_concurrency", 1)

        backoff = rl.get("backoff", {})
        self.backoff_initial = backoff.get("initial_ms", 500)
        self.backoff_max = backoff.get("max_ms", 4000)
        self.backoff_multiplier = backoff.get("multiplier", 2.0)

    def sleep_with_jitter(self):
        """Apply intentional delay + jitter between requests."""
        delay = self.delay_ms + random.uniform(0, self.jitter_ms)
        time.sleep(delay / 1000)

    def backoff(self, attempt):
        """Exponential backoff for 429/5xx errors."""
        delay = min(
            self.backoff_initial * (self.backoff_multiplier ** attempt),
            self.backoff_max
        )
        time.sleep(delay / 1000)
