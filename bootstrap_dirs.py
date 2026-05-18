# bootstrap_dirs.py

import os
import yaml
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()



def load_config():
    with open("configs/synthetic.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    raw_output = cfg["output_dir"]
    resolved = os.path.expandvars(raw_output)
    out_dir = Path(resolved)

    print(f"[bootstrap] output_dir (raw): {raw_output}")
    print(f"[bootstrap] output_dir (resolved): {out_dir}")

    # Create root directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    subdirs = ["raw", "validated", "deduped", "rejected", "manifests"]

    for sub in subdirs:
        p = out_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        print(f"[bootstrap] created: {p}")

    print("[bootstrap] directory structure ready.")


if __name__ == "__main__":
    main()
