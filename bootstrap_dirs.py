# bootstrap_dirs.py

from __future__ import annotations

from slm_synth.paths import load_yaml_config, resolve_output_dir


def main() -> None:
    cfg = load_yaml_config("configs/synthetic.yaml")

    raw_output = cfg["output_dir"]
    out_dir = resolve_output_dir(cfg)

    print(f"[bootstrap] output_dir (raw): {raw_output}")
    print(f"[bootstrap] output_dir (resolved): {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    subdirs = ["raw", "validated", "deduped", "rejected", "manifests"]
    for sub in subdirs:
        p = out_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        print(f"[bootstrap] created: {p}")

    print("[bootstrap] directory structure ready.")


if __name__ == "__main__":
    main()
