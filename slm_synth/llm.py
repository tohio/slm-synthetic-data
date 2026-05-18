# slm_synth/llm.py

import os
import json
import re
import time
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import httpx


def extract_json_array(text: str) -> str:
    """
    Extract the first JSON array from a messy LLM output.
    """
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in model output")
    return match.group(0)


class LLMBackend:
    """
    Groq backend that supports:
    - expect_array=True
    - expected_length=N
    - JSON array extraction
    - batch validation
    - retries for malformed output
    """

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float, top_p: float):
        if provider != "groq":
            raise ValueError(f"Unsupported provider: {provider}. Only 'groq' is supported.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")

        # Use a clean httpx client (avoids Groq SDK proxy bug)
        http_client = httpx.Client(timeout=60.0)

        self.client = Groq(api_key=api_key, http_client=http_client)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    # ------------------------------------------------------------
    # Core LLM call
    # ------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message["content"]

    # ------------------------------------------------------------
    # Public API used by generators
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
            - extract array from messy output
            - validate array length
        Otherwise:
            - return raw string
        """

        for attempt in range(max_retries):
            raw = self._call_llm(prompt)

            if not expect_array:
                return raw

            try:
                # Try direct JSON parse
                arr = json.loads(raw)
                if not isinstance(arr, list):
                    raise ValueError("Model returned non-list JSON")
            except Exception:
                # Try extracting array from messy output
                try:
                    cleaned = extract_json_array(raw)
                    arr = json.loads(cleaned)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to parse JSON array: {e}")
                    time.sleep(0.5 * (attempt + 1))
                    continue

            # Validate batch size
            if expected_length is not None and len(arr) != expected_length:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Model returned wrong batch size: expected {expected_length}, got {len(arr)}"
                    )
                time.sleep(0.5 * (attempt + 1))
                continue

            return arr

        raise RuntimeError("Unreachable: exceeded retry loop")