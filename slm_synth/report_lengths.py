"""Compatibility wrapper for slm_synth.pretrain.report_lengths."""

from slm_synth.pretrain.report_lengths import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("slm_synth.pretrain.report_lengths", run_name="__main__")
