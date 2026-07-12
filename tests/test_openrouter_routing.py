from types import ModuleType, SimpleNamespace
import sys

import pytest

OPENROUTER_ENV_KEYS = (
    "OPENROUTER_ROUTING_MODE",
    "OPENROUTER_PROVIDER",
    "OPENROUTER_PROVIDER_ORDER",
    "OPENROUTER_PROVIDER_ONLY",
    "OPENROUTER_PROVIDER_IGNORE",
    "OPENROUTER_PROVIDER_SORT",
)


@pytest.fixture(autouse=True)
def clear_openrouter_env(monkeypatch):
    for key in OPENROUTER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from slm_synth.llm import (
    LLMBackend,
    OpenRouterRoutingPolicy,
    resolve_openrouter_routing_policy,
)


def test_openrouter_routing_policy_auto_allows_default_fallbacks():
    policy = resolve_openrouter_routing_policy(mode="auto", provider=None)

    assert policy.provider_preferences(require_parameters=True, allow_fallbacks=True) == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert policy.metadata(allow_fallbacks=True) == {
        "routing_mode": "auto",
        "requested_provider": None,
        "provider_order": [],
        "provider_only": [],
        "provider_ignore": [],
        "provider_sort": None,
        "allow_fallbacks": True,
    }


def test_openrouter_routing_policy_prefer_orders_provider_and_allows_fallbacks():
    policy = resolve_openrouter_routing_policy(mode="prefer", provider="deepinfra")

    assert policy.provider_preferences(require_parameters=True, allow_fallbacks=False) == {
        "require_parameters": True,
        "order": ["deepinfra"],
        "allow_fallbacks": True,
    }


def test_openrouter_routing_policy_strict_allows_only_requested_provider():
    policy = resolve_openrouter_routing_policy(mode="strict", provider="deepinfra")

    assert policy.provider_preferences(require_parameters=True, allow_fallbacks=True) == {
        "require_parameters": True,
        "only": ["deepinfra"],
        "allow_fallbacks": False,
    }


def test_openrouter_routing_policy_auto_supports_order_ignore_and_sort():
    policy = resolve_openrouter_routing_policy(
        mode="auto",
        provider=None,
        provider_order="Venice,Alibaba",
        provider_ignore="Baidu",
        provider_sort="throughput",
    )

    assert policy.provider_preferences(require_parameters=True, allow_fallbacks=True) == {
        "require_parameters": True,
        "order": ["Venice", "Alibaba"],
        "allow_fallbacks": True,
        "ignore": ["Baidu"],
        "sort": "throughput",
    }
    assert policy.metadata(allow_fallbacks=True) == {
        "routing_mode": "auto",
        "requested_provider": None,
        "provider_order": ["Venice", "Alibaba"],
        "provider_only": [],
        "provider_ignore": ["Baidu"],
        "provider_sort": "throughput",
        "allow_fallbacks": True,
    }


def test_openrouter_routing_policy_auto_supports_only_provider_list():
    policy = resolve_openrouter_routing_policy(
        mode="auto",
        provider=None,
        provider_only="Venice,Alibaba",
    )

    assert policy.provider_preferences(require_parameters=True, allow_fallbacks=True) == {
        "require_parameters": True,
        "only": ["Venice", "Alibaba"],
        "allow_fallbacks": True,
    }


def test_openrouter_routing_policy_rejects_order_with_provider_mode():
    with pytest.raises(ValueError, match="OPENROUTER_PROVIDER_ORDER"):
        resolve_openrouter_routing_policy(
            mode="strict",
            provider="Venice",
            provider_order="Alibaba",
        )


@pytest.mark.parametrize("mode", ["prefer", "strict"])
def test_openrouter_routing_policy_requires_provider_for_provider_modes(mode):
    with pytest.raises(ValueError, match="OPENROUTER_PROVIDER is required"):
        resolve_openrouter_routing_policy(mode=mode, provider=None)


def test_openrouter_routing_policy_rejects_provider_with_auto_mode():
    with pytest.raises(ValueError, match="OPENROUTER_PROVIDER requires"):
        resolve_openrouter_routing_policy(mode="auto", provider="deepinfra")


def test_openrouter_structured_completion_uses_routing_policy():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))])

    backend = LLMBackend.__new__(LLMBackend)
    backend.provider = "openrouter"
    backend.model = "deepseek/deepseek-v4-flash"
    backend.max_tokens = 1024
    backend.temperature = 0.2
    backend.top_p = 0.95
    backend.require_parameters = True
    backend.allow_fallbacks = True
    backend.openrouter_routing_policy = OpenRouterRoutingPolicy(
        mode="strict",
        requested_provider="deepinfra",
    )
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    backend._create_structured_completion("prompt", {"type": "object"}, "schema")

    assert calls[0]["extra_body"] == {
        "provider": {
            "require_parameters": True,
            "only": ["deepinfra"],
            "allow_fallbacks": False,
        }
    }


def test_openrouter_routing_metadata_is_available_for_manifests():
    backend = LLMBackend.__new__(LLMBackend)
    backend.require_parameters = True
    backend.allow_fallbacks = True
    backend.openrouter_routing_policy = OpenRouterRoutingPolicy(
        mode="prefer",
        requested_provider="deepinfra",
    )

    assert backend._routing_metadata() == {
        "routing_mode": "prefer",
        "requested_provider": "deepinfra",
        "provider_order": [],
        "provider_only": [],
        "provider_ignore": [],
        "provider_sort": None,
        "allow_fallbacks": True,
    }
