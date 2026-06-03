from distill.generation.hosted_backend import HostedBackendConfig


def test_hosted_backend_config_openrouter_defaults() -> None:
    config = HostedBackendConfig(provider_name='openrouter')
    assert config.json_mode is True
    assert config.require_parameters is True
    assert config.allow_fallbacks is False
    assert config.request_timeout_seconds == 300.0


def test_hosted_backend_config_groq_defaults() -> None:
    config = HostedBackendConfig(provider_name='groq')
    assert config.json_mode is True
    assert config.request_timeout_seconds == 300.0
