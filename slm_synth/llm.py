import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMBackend:
    """
    Groq-backed LLM client for scalable synthetic generation.

    The preferred contract is a top-level JSON object:
        {"items": [{...}, {...}]}

    Bare arrays are still accepted for backward compatibility, but prompts should
    use the object contract so Groq JSON object mode can be enabled.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        *,
        json_mode: bool = True,
        service_tier: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_request_retries: int = 3,
        retry_sleep_seconds: float = 0.5,
    ):
        if provider != "groq":
            raise ValueError(f"Unsupported provider: {provider}")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment.")

        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.json_mode = bool(json_mode)
        self.service_tier = service_tier
        self.request_timeout = request_timeout
        self.max_request_retries = int(max_request_retries)
        self.retry_sleep_seconds = float(retry_sleep_seconds)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=request_timeout,
        )

    def clone_for(
        self,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        json_mode: Optional[bool] = None,
        service_tier: Optional[str] = None,
    ) -> "LLMBackend":
        """Create a same-provider client with per-signal overrides."""
        return LLMBackend(
            provider="groq",
            model=model or self.model,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            json_mode=self.json_mode if json_mode is None else json_mode,
            service_tier=self.service_tier if service_tier is None else service_tier,
            request_timeout=self.request_timeout,
            max_request_retries=self.max_request_retries,
            retry_sleep_seconds=self.retry_sleep_seconds,
        )

    def _clean(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        return text

    def _extract_json_candidate(self, raw: str) -> str:
        raw = self._clean(raw)
        if raw.startswith("{") or raw.startswith("["):
            return raw

        # Prefer object contract, fall back to array contract.
        object_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if object_match:
            return object_match.group(0)
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if array_match:
            return array_match.group(0)
        raise ValueError(f"Model did not return JSON: {raw[:2000]}")

    def _parse_items(self, raw: str, batch_size: int) -> List[Dict[str, Any]]:
        candidate = self._extract_json_candidate(raw)
        parsed = json.loads(candidate)

        if isinstance(parsed, dict):
            objs = parsed.get("items")
        elif isinstance(parsed, list):
            objs = parsed
        else:
            objs = None

        if not isinstance(objs, list):
            raise ValueError("Expected a JSON object with an 'items' array")

        if len(objs) != batch_size:
            raise ValueError(f"Expected {batch_size} items, got {len(objs)}")

        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError("Every generated item must be a JSON object")

        return objs

    def _create_completion(self, prompt: str):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only synthetic data generator. "
                        "Return exactly one valid JSON object and no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Groq-compatible OpenAI clients may accept service_tier directly. If
        # the installed SDK rejects it, retry once using extra_body.
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier

        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:
            tier = kwargs.pop("service_tier", None)
            if tier:
                kwargs["extra_body"] = {"service_tier": tier}
                return self.client.chat.completions.create(**kwargs)
            raise

    def generate_batch(self, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_request_retries + 1):
            try:
                response = self._create_completion(prompt)
                raw = (response.choices[0].message.content or "").strip()
                return self._parse_items(raw, batch_size)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_request_retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)

        raise RuntimeError(f"LLM batch failed after {self.max_request_retries} attempts: {last_error}")
