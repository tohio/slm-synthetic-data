import os
import requests


class LLMBackend:
    """Unified interface for Groq or HF text generation."""

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float, top_p: float):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        if provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.url = "https://api.groq.com/openai/v1/chat/completions"

        elif provider == "hf":
            self.api_key = os.getenv("HF_TOKEN")
            self.url = f"https://api-inference.huggingface.co/models/{model}"

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str) -> str:
        if self.provider == "groq":
            return self._generate_groq(prompt)
        return self._generate_hf(prompt)

    def _generate_groq(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(self.url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _generate_hf(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": prompt}
        r = requests.post(self.url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()[0]["generated_text"]
