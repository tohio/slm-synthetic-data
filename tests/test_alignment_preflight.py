import copy

import pytest

from slm_synth.alignment_preflight import (
    _validate_declared_sources,
    preflight_all_inventories,
    preflight_dpo_inventory,
    preflight_sft_inventory,
)
from slm_synth.dpo.source_catalog import DPO_SOURCE_CATALOG
from slm_synth.dpo.spec_builders import DPO_SPEC_CAPACITIES, build_complete_inventory as build_dpo_inventory
from slm_synth.sft.source_catalog import SFT_SOURCE_CATALOG
from slm_synth.sft.spec_builders import SFT_SPEC_CAPACITIES, build_complete_inventory as build_sft_inventory


def test_complete_alignment_inventories_pass_preflight():
    report = preflight_all_inventories()
    assert report["status"] == "clean"
    assert report["sft"]["total_capacity"] == 60
    assert report["dpo"]["total_capacity"] == 90
    assert report["sft"]["capacity_by_group"] == dict(sorted(SFT_SPEC_CAPACITIES.items()))
    assert report["dpo"]["capacity_by_group"] == dict(sorted(DPO_SPEC_CAPACITIES.items()))


def test_sft_inventory_has_declared_axis_coverage():
    report = preflight_sft_inventory()
    assert report["interaction_modes"]["multi_turn"] > 0
    assert report["interaction_modes"]["system_conditioned"] > 0
    assert set(report["context_modes"]) == {"self_contained", "supplied_passage", "long_document", "multi_document"}
    assert {"free_text", "concise", "structured_json", "table", "exact_constraints", "code"} <= set(report["output_modes"])


def test_dpo_inventory_is_independent_and_covers_tool_preferences():
    report = preflight_dpo_inventory()
    assert report["output_modes"]["tool_call"] > 0
    assert report["interaction_modes"]["tool_mediated"] > 0
    sft_tasks = {(spec["instruction"], repr(spec.get("variables"))) for spec in build_sft_inventory()}
    assert all((spec["instruction"], repr(spec.get("variables"))) not in sft_tasks for spec in build_dpo_inventory())


def test_preflight_rejects_renamed_source_key():
    catalog = copy.deepcopy(SFT_SOURCE_CATALOG)
    first_family, second_family = sorted(catalog)[:2]
    catalog[second_family][0]["source_key"] = catalog[first_family][0]["source_key"]
    with pytest.raises(ValueError, match="source_key.*repeats"):
        _validate_declared_sources(catalog, kind="SFT")


def test_preflight_rejects_number_only_variation():
    catalog = {
        "example": tuple(
            {
                "source_key": f"different_task_{letter}",
                "instruction": f"Calculate the total for {number} boxes.",
                "variables": {"boxes": number},
                "metadata": {"template_family": f"template_{letter}"},
            }
            for letter, number in zip("abcde", (10, 20, 30, 40, 50), strict=True)
        )
    }
    with pytest.raises(ValueError, match="differ only by numbers"):
        _validate_declared_sources(catalog, kind="test")
