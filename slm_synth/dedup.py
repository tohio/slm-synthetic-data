"""Compatibility wrapper for slm_synth.pretrain.dedup."""

from slm_synth.pretrain.dedup import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("slm_synth.pretrain.dedup", run_name="__main__")
