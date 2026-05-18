import json
from pathlib import Path


class JSONLWriter:
    """
    Writes one JSON object per line.
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.output_path, "a", encoding="utf-8")

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()
