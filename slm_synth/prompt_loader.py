import yaml
from pathlib import Path


def load_prompt(path: str):
    """Load a YAML prompt file and return a dict."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
