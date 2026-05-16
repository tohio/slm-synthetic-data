import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file


def main(dedup_dir: str):
    dedup_dir = Path(dedup_dir)
    repo = f"{os.getenv('HF_USERNAME')}/{os.getenv('HF_REPO')}"
    token = os.getenv("HF_TOKEN")

    api = HfApi()

    try:
        create_repo(repo, token=token, exist_ok=True)
    except Exception:
        pass

    for file in dedup_dir.glob("*.jsonl"):
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=repo,
            token=token,
        )


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
