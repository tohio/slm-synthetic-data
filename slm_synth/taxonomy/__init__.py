"""Shared taxonomy labels for synthetic alignment datasets."""

from slm_synth.taxonomy.categories import CATEGORIES, validate_category
from slm_synth.taxonomy.difficulties import MAX_DIFFICULTY, MIN_DIFFICULTY, validate_difficulty
from slm_synth.taxonomy.eval_families import EVAL_FAMILIES, validate_eval_family
from slm_synth.taxonomy.failure_modes import FAILURE_MODES, validate_failure_mode
from slm_synth.taxonomy.metadata import validate_metadata
from slm_synth.taxonomy.template_families import validate_template_family
from slm_synth.taxonomy.context_modes import CONTEXT_MODES, validate_context_mode
from slm_synth.taxonomy.interaction_modes import INTERACTION_MODES, validate_interaction_modes
from slm_synth.taxonomy.output_modes import OUTPUT_MODES, validate_output_mode
from slm_synth.taxonomy.preference_dimensions import PREFERENCE_DIMENSIONS, validate_preference_dimension
from slm_synth.taxonomy.task_families import TASK_FAMILIES, validate_task_family
from slm_synth.taxonomy.alignment_metadata import validate_alignment_metadata

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
    "CONTEXT_MODES", "INTERACTION_MODES", "OUTPUT_MODES", "PREFERENCE_DIMENSIONS", "TASK_FAMILIES",
    "validate_alignment_metadata", "validate_context_mode", "validate_interaction_modes",
    "validate_output_mode", "validate_preference_dimension", "validate_task_family",
]
