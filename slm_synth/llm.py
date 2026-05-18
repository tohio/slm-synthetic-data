import json
import time
from typing import Optional
from openai import OpenAI


class LLMBackend:
    """
    JSONL-compatible LLM backend.
    - One prompt → one JSON object
    - No batching
    - No array parsing
    """

    def __init__(self, provider: str, model: str, max_tokens: int,
                 temperature: float, top_p: float, api_key: Optional[str] = None):

        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Groq uses OpenAI-compatible client
        self.client = OpenAI(api_key=api_key)

    # ---------------------------------------------------------
    # Clean model output (remove markdown, code fences, chatter)
    # ---------------------------------------------------------
    def _clean(self, text: str) -> str:
        text = text.strip()

        # Remove ```json or ``` fences
        if text.startswith("```"):
            text = text.split("```", 1)[-1].strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()

        return text

    # ---------------------------------------------------------
    # Core JSONL generation: one prompt → one JSON object
    # ---------------------------------------------------------
    def generate_one(self, prompt: str, retries: int = 3, delay: float = 0.5):
        """
        Sends a single prompt and expects a single JSON object.
        Retries on malformed JSON.
        """

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                raw = response.choices[0].message.content
                cleaned = self._clean(raw)

                # Parse JSON object
                obj = json.loads(cleaned)
                if not isinstance(obj, dict):
                    raise ValueError("Model did not return a JSON object")

                return obj

            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to parse JSON object: {e}")

                time.sleep(delay * (attempt + 1))

        raise RuntimeError("Unreachable: JSON parsing loop exited unexpectedly")
