"""Attribute unchanged detector evidence to provisional core/background regions.

The detector runs on unmasked source pixels. Scope attribution never removes
stored candidates, changes their contrast scores or asserts a defect cause.
"""
from __future__ import annotations

from .core import label_counts_in_box


def attribute_candidates(local, proposal, labels):
    shape = proposal["source_shape_yx"]
    step = proposal["native_pixels_per_label_cell"]
    if local["source_shape_yx"] != shape:
        raise ValueError("Core proposal and local source geometry disagree")
    attributed = []
    for candidate in local["candidates"]:
        box = candidate["bbox_level0_xyxy"]
        context = candidate["context_bbox_level0_xyxy"]
        counts = label_counts_in_box(labels, step, shape, box)
        surrounding = label_counts_in_box(labels, step, shape, context)
        area = (box[2] - box[0]) * (box[3] - box[1])
        context_area = (context[2] - context[0]) * (context[3] - context[1])
        core = (counts["core_dim_or_space"] + counts["tissue_support"]) / area
        core_context = (surrounding["core_dim_or_space"] + surrounding["tissue_support"]) / context_area
        background = counts["nearby_background"] / area
        neighbour = counts["neighbour_guard"] / area
        if core == 1 and core_context == 1:
            scope = "core_interior"
        elif core > 0 or core_context > 0:
            scope = "core_boundary_or_mixed_context"
        elif background == 1:
            scope = "nearby_background"
        elif neighbour > 0:
            scope = "neighbour_or_mixed_context"
        else:
            scope = "outside_or_unassigned"
        attributed.append({"candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
                           "bbox_level0_xyxy": box, "scope": scope,
                           "core_envelope_fraction": core, "core_context_fraction": core_context,
                           "tissue_support_fraction": counts["tissue_support"] / area,
                           "nearby_background_fraction": background, "neighbour_guard_fraction": neighbour,
                           "native_log_contrast": candidate["native_log_contrast"],
                           "target_identity_status": proposal["target_identity_status"],
                           "artifact_cause_verified": False, "quality_decision": None})
    scopes = ["core_interior", "core_boundary_or_mixed_context", "nearby_background",
              "neighbour_or_mixed_context", "outside_or_unassigned"]
    coverage = local["coverage"]
    full_source = (local["target_region_level0_xyxy"] == [0, 0, shape[1], shape[0]]
                   and coverage["detector_evaluated_area_fraction"] == 1)
    areas = proposal["label_areas_native_px"]
    regions_present = areas["core_dim_or_space"] + areas["tissue_support"] > 0 and areas["nearby_background"] > 0
    return {"schema": "fluoinspect.core-candidate-attribution.v1",
            "candidate_count": len(attributed), "candidates": attributed,
            "counts_by_scope": {scope: sum(row["scope"] == scope for row in attributed) for scope in scopes},
            "same_source_context_and_detector_lattice": True,
            "source_was_masked_before_detection": False, "detector_settings_changed": False,
            "all_original_candidates_retained": True,
            "target_identity_status": proposal["target_identity_status"],
            "core_boundary_reviewed": False,
            "core_and_background_detector_footprint_fraction": 1.0 if full_source and regions_present else None,
            "partial_source_region_coverage_established": False if not full_source else None,
            "model_inspected_area_fraction": None, "human_inspected_area_fraction": None,
            "note": "Region attribution isolates scope effects using identical measurements; it is not a rerun on artificially zeroed or cropped pixels.",
            "quality_decision": None}
