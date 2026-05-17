# slm_synth/llm.py

import os
from groq import Groq


class LLMBackend:
    """
    Minimal Groq-only backend for chat completions.
    No batching, no JSON enforcement, no OpenAI compatibility layer.
    """

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float, top_p: float):
        if provider != "groq":
            raise ValueError(f"Unsupported provider: {provider}. Only 'groq' is supported.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    # ------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        Single prompt → single completion.
        Generators handle batching, so this stays minimal.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return response.choices[0].message["content"]
