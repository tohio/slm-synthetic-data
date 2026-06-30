"""Compatibility wrapper for slm_synth.pretrain.report_artifacts."""

from slm_synth.pretrain.report_artifacts import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("slm_synth.pretrain.report_artifacts", run_name="__main__")
