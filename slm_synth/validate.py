"""Compatibility wrapper for slm_synth.pretrain.validate."""

from slm_synth.pretrain.validate import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("slm_synth.pretrain.validate", run_name="__main__")
