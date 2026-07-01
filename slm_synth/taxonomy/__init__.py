"""Shared taxonomy labels for synthetic alignment datasets."""

from slm_synth.taxonomy.categories import CATEGORIES, validate_category
from slm_synth.taxonomy.difficulties import MAX_DIFFICULTY, MIN_DIFFICULTY, validate_difficulty
from slm_synth.taxonomy.eval_families import EVAL_FAMILIES, validate_eval_family
from slm_synth.taxonomy.failure_modes import FAILURE_MODES, validate_failure_mode
from slm_synth.taxonomy.metadata import validate_metadata
from slm_synth.taxonomy.template_families import validate_template_family

__all__ = [
    "CATEGORIES",
    "EVAL_FAMILIES",
    "FAILURE_MODES",
    "MAX_DIFFICULTY",
    "MIN_DIFFICULTY",
    "validate_category",
    "validate_difficulty",
    "validate_eval_family",
    "validate_failure_mode",
    "validate_metadata",
    "validate_template_family",
]
