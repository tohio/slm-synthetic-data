import json
import re
import time
from typing import Any, Optional


class LLMBackend:
    def __init__(
        self,
        provider: Any,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    # ------------------------------------------------------------
    # Low-level raw call (provider-specific)
    # ------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured provider.

        The provider object must expose a generate(...) method that returns
        the raw string response from the model.
        """
        if not hasattr(self.provider, "generate"):
            raise TypeError(
                "LLM provider must expose a generate(...) method. "
                f"Got provider={self.provider!r}."
            )

        return self.provider.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    # ------------------------------------------------------------
    # Extract JSON array from messy output
    # ------------------------------------------------------------
    @staticmethod
    def _extract_json_array(text: str) -> str:
        """
        Extract the first JSON array from model output.

        Handles responses wrapped in prose or markdown fences, such as:

            ```json
            [{...}, {...}]
            ```

        This intentionally extracts arrays only, because generation expects
        batched JSONL-style records to be returned as a JSON array.
        """
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in model output")
        return match.group(0)

    # ------------------------------------------------------------
    # High-level generate() with optional batch enforcement
    # ------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        expect_array: bool = False,
        expected_length: Optional[int] = None,
        max_retries: int = 3,
    ):
        """
        Generate text or a parsed JSON array.

        If expect_array=True:
            - enforce JSON array output
            - extract array if wrapped in prose or markdown
            - validate array length if expected_length is provided

        Otherwise:
            - return the raw model string
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            raw = self._call_llm(prompt)

            if not expect_array:
                return raw

            try:
                arr = json.loads(raw)

                if not isinstance(arr, list):
                    raise ValueError("Model returned non-array JSON")

            except Exception as exc:
                last_error = exc

                try:
                    cleaned = self._extract_json_array(raw)
                    arr = json.loads(cleaned)

                    if not isinstance(arr, list):
                        raise ValueError("Extracted JSON was not an array")

                except Exception as extract_exc:
                    last_error = extract_exc
                    time.sleep(0.2 * (attempt + 1))
                    continue

            if expected_length is not None and len(arr) != expected_length:
                last_error = ValueError(
                    f"Expected {expected_length} records, got {len(arr)}"
                )
                time.sleep(0.2 * (attempt + 1))
                continue

            return arr

        raise RuntimeError(
            "LLM failed to produce a valid JSON array after retries"
        ) from last_error
