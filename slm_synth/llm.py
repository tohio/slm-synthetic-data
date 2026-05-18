import json
import time
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class LLMBackend:
    """
    JSONL-compatible LLM backend for Groq.
    One prompt → one JSON object.
    """

    def __init__(self, provider: str, model: str, max_tokens: int,
                 temperature: float, top_p: float):

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # IMPORTANT: Groq requires base_url override
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def _clean(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 1)[-1].strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        return text

    def generate_batch(self, prompt: str, batch_size: int):
        """
        Generate a batch of JSON objects using a single LLM call.
        The model must return a JSON array of length batch_size.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,  # one response containing a JSON array
        )

        raw = response.choices[0].message.content.strip()

        # Parse the JSON array
        objs = json.loads(raw)

        if not isinstance(objs, list):
            raise ValueError("Expected a JSON array from batched prompt")

        if len(objs) != batch_size:
            raise ValueError(f"Expected {batch_size} items, got {len(objs)}")

        return objs

