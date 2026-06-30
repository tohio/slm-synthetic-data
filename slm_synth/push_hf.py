"""Compatibility wrapper for slm_synth.pretrain.push_hf."""

from slm_synth.pretrain.push_hf import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("slm_synth.pretrain.push_hf", run_name="__main__")
