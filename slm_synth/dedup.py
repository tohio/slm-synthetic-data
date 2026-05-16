import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH


def minhash(text: str):
    mh = MinHash(num_perm=128)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh


def dedup_file(path: Path, out_path: Path, threshold: float):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    seen = {}

    with open(path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            obj = json.loads(line)
            text = json.dumps(obj, ensure_ascii=False)

            mh = minhash(text)

            dup = False
            for key, existing in seen.items():
                if existing.jaccard(mh) >= threshold:
                    dup = True
                    break

            if not dup:
                key = len(seen)
                seen[key] = mh
                out.write(line)


def main(validated_dir: str):
    validated_dir = Path(validated_dir)
    deduped = validated_dir.parent / "deduped"
    deduped.mkdir(exist_ok=True)

    for file in validated_dir.glob("*.jsonl"):
        out_file = deduped / file.name
        dedup_file(file, out_file, threshold=0.85)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
