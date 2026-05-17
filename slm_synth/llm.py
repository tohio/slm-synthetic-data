import json
import re
import time
import random
from typing import Optional


class LLMBackend:
    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float, top_p: float):
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
        Replace this with your actual provider call.
        Must return a raw string from the model.
        """
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
        Extract the first JSON array from the text.
        Handles cases where the model adds prose, markdown, etc.
        """
        match = re.search(r"

\[\s*{.*}\s*\]

", text, flags=re.DOTALL)
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
        If expect_array=True:
            - enforce JSON array output
            - extract array if wrapped in junk
            - validate array length if expected_length is provided
        Otherwise:
            - return raw string
        """
        for attempt in range(max_retries):
            raw = self._call_llm(prompt)

            if not expect_array:
                return raw

            try:
                # Try direct parse
                arr = json.loads(raw)

                if not isinstance(arr, list):
                    raise ValueError("Model returned non-array JSON")

            except Exception:
                # Try extracting array from messy output
                try:
                    cleaned = self._extract_json_array(raw)
                    arr = json.loads(cleaned)
                except Exception:
                    # Retry
                    time.sleep(0.2 * (attempt + 1))
                    continue

            # Validate batch size
            if expected_length is not None and len(arr) != expected_length:
                # Retry if wrong length
                time.sleep(0.2 * (attempt + 1))
                continue

            return arr

        raise RuntimeError("LLM failed to produce a valid JSON array after retries")