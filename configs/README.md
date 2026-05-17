# configs

Configuration files that define how a synthetic‑data run behaves. Each YAML file
specifies dataset size, signal families, sampling parameters, validation rules,
and output paths. The pipeline reads one config per run and executes the full
generation → validation → dedup → export workflow based on these settings.
