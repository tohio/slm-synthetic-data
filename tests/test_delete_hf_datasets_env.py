from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_delete_module():
    module_path = Path("scripts/delete_hf_datasets.py")
    spec = importlib.util.spec_from_file_location("delete_hf_datasets_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delete_hf_datasets_loads_hf_token_from_dotenv(tmp_path, monkeypatch) -> None:
    module = _load_delete_module()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=hf_test_token\n", encoding="utf-8")

    loaded = module._load_dotenv_if_needed(dotenv)

    assert loaded is True
    assert os.environ["HF_TOKEN"] == "hf_test_token"


def test_delete_hf_datasets_loads_quoted_hub_token_from_dotenv(tmp_path, monkeypatch) -> None:
    module = _load_delete_module()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    dotenv = tmp_path / ".env"
    dotenv.write_text('HUGGINGFACE_HUB_TOKEN="hf_hub_test_token"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert module._get_hf_token() == "hf_hub_test_token"


def test_delete_hf_datasets_does_not_override_exported_token(tmp_path, monkeypatch) -> None:
    module = _load_delete_module()
    monkeypatch.setenv("HF_TOKEN", "hf_exported_token")

    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=hf_dotenv_token\n", encoding="utf-8")

    module._load_dotenv_if_needed(dotenv)

    assert os.environ["HF_TOKEN"] == "hf_exported_token"
