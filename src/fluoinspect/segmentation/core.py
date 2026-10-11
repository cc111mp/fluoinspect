"""Bounded traditional core proposals with dark-retaining outer envelopes.

Labels describe coarse, provisional regions. Source intensities are untouched.
Geometry can be wrong on fragmented cores, shadows or neighbours; selection
remains unreviewed or unresolved and is never a quality decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from ..io.native import MAX_NATIVE_PIXELS
from ..measurements.brightness import _block_mean, _intensity_region

ENGINE_ID = "core_envelope_proposals_v1"
LABELS = {"outside": 0, "core_dim_or_space": 1, "tissue_support": 2,
          "nearby_background": 3, "neighbour_guard": 4}


@dataclass(frozen=True)
class CoreConfig:
    max_overview_dimension: int = 512
    closing_fraction: float = 0.035
    fragment_gap_fraction: float = 0.10
    min_object_area_fraction: float = 0.008
    max_envelope_area_fraction: float = 0.65
    max_axis_ratio: float = 1.8
    max_flat_edge_perimeter_fraction: float = 0.20
    max_center_distance_fraction: float = 0.35
    ambiguity_margin: float = 0.10
    background_ring_fraction: float = 0.08
    neighbour_guard_fraction: float = 0.015
    envelope_padding_cells: int = 1
    max_objects: int = 32

    def __post_init__(self):
        for name, low, high in (("max_overview_dimension", 128, 1024),
                                ("envelope_padding_cells", 0, 8), ("max_objects", 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        for name, low, high in (("closing_fraction", 0, 0.15), ("fragment_gap_fraction", 0, 0.25),
                                ("min_object_area_fraction", 0, 0.1), ("max_envelope_area_fraction", 0, 1),
                                ("max_axis_ratio", 1, 4), ("max_center_distance_fraction", 0, 1),
                                ("max_flat_edge_perimeter_fraction", 0, 0.5),
                                ("ambiguity_margin", 0, 1), ("background_ring_fraction", 0, 0.3),
                                ("neighbour_guard_fraction", 0, 0.1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"Invalid {name}")
        if self.min_object_area_fraction >= self.max_envelope_area_fraction:
            raise ValueError("Object area bounds disagree")

    def to_dict(self):
        return asdict(self)


def _kernel(radius):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _geometry(points, shape):
    hull = cv2.convexHull(points.astype(np.int32)).reshape(-1, 2)
    x0, y0 = hull.min(axis=0)
    x1, y1 = hull.max(axis=0) + 1
    size = (int(x1 - x0), int(y1 - y0))
    edges = np.linalg.norm(np.roll(hull.astype(float), -1, axis=0) - hull, axis=1)
    return {"hull": hull, "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "axis_ratio": max(size) / min(size),
            "area": float(cv2.contourArea(hull)) + 1,
            "longest_flat_edge_perimeter_fraction": float(edges.max() / max(1e-9, edges.sum())),
            "center": hull.mean(axis=0),
            "touches_border": bool(x0 == 0 or y0 == 0 or x1 == shape[1] or y1 == shape[0])}


def _box_gap(first, second):
    dx = max(0, first[0] - second[2], second[0] - first[2])
    dy = max(0, first[1] - second[3], second[1] - first[3])
    return math.hypot(dx, dy)


def _native_box(box, step, shape):
    return [box[0] * step, box[1] * step, min(shape[1], box[2] * step), min(shape[0], box[3] * step)]


def label_counts_in_box(labels, step, shape_yx, box):
    """Exact native-pixel areas of the recorded piecewise-constant coarse labels."""
    from ..investigation.session import validate_box
    if (type(step) is not int or step < 1 or not isinstance(labels, np.ndarray)
            or labels.dtype != np.uint8
            or list(labels.shape) != [math.ceil(v / step) for v in shape_yx]):
        raise ValueError("Region label geometry disagrees with the source")
    x0, y0, x1, y1 = validate_box(box, shape_yx)
    ix0, iy0 = x0 // step, y0 // step
    ix1, iy1 = math.ceil(x1 / step), math.ceil(y1 / step)
    xs = np.arange(ix0, ix1) * step
    ys = np.arange(iy0, iy1) * step
    widths = np.maximum(0, np.minimum(xs + step, x1) - np.maximum(xs, x0))
    heights = np.maximum(0, np.minimum(ys + step, y1) - np.maximum(ys, y0))
    weights = heights[:, None] * widths[None, :]
    selected = labels[iy0:iy1, ix0:ix1]
    return {key: int(weights[selected == value].sum()) for key, value in LABELS.items()}


def _label_bbox(mask, step, native_shape):
    yy, xx = np.where(mask)
    return _native_box([int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1], step, native_shape) if yy.size else None


def propose_core(native, *, config=None, target_point_native_xy=None):
    config = CoreConfig() if config is None else config
    if not isinstance(config, CoreConfig):
        raise ValueError("A CoreConfig is required")
    if (not isinstance(native, np.ndarray) or isinstance(native, np.ma.MaskedArray)
            or native.ndim != 2 or native.dtype.kind != "u"
            or native.dtype.itemsize != 2 or not native.size or native.size > MAX_NATIVE_PIXELS):
        raise ValueError("Expected bounded native uint16 YX pixels")
    h, w = native.shape
    if target_point_native_xy is not None and (
            not isinstance(target_point_native_xy, (list, tuple)) or len(target_point_native_xy) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in target_point_native_xy)
            or not 0 <= target_point_native_xy[0] < w or not 0 <= target_point_native_xy[1] < h):
        raise ValueError("Target point must lie inside the source")
    target = [w / 2, h / 2] if target_point_native_xy is None else list(target_point_native_xy)
    step = max(1, math.ceil(max(h, w) / config.max_overview_dimension))
    means = _block_mean(native, step)
    support = _intensity_region(np.log1p(means))
    span = min(means.shape)
    radius = max(1, round(span * config.closing_fraction / 2))
    closed = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(radius))
    count, components, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    indices = [i for i in range(1, count)
               if stats[i, cv2.CC_STAT_AREA] >= config.min_object_area_fraction * means.size]
    indices.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    omitted = max(0, len(indices) - config.max_objects)
    groups = []
    for index in indices[:config.max_objects]:
        contours, _ = cv2.findContours((components == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = np.concatenate(contours).reshape(-1, 2)
        if len(points) < 3:
            continue
        groups.append({"members": [int(index)], **_geometry(points, means.shape)})
    # Fragment grouping only proposes a shared envelope. It does not restore a core.
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(groups):
            if first["touches_border"]:
                continue
            for j in range(i + 1, len(groups)):
                second = groups[j]
                if second["touches_border"] or _box_gap(first["bbox"], second["bbox"]) > span * config.fragment_gap_fraction:
                    continue
                joined = _geometry(np.concatenate([first["hull"], second["hull"]]), means.shape)
                if (joined["axis_ratio"] <= config.max_axis_ratio
                        and joined["area"] <= 1.35 * (first["area"] + second["area"])
                        and joined["area"] <= config.max_envelope_area_fraction * means.size):
                    groups[i] = {"members": first["members"] + second["members"], **joined}
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    target_cell = np.asarray(target) / step
    for group in groups:
        group["center_distance"] = float(np.linalg.norm(group["center"] - target_cell) / max(1, span))
        group["ranking_score"] = group["center_distance"] + 0.7 * group["touches_border"]
    groups.sort(key=lambda group: group["ranking_score"])
    reasons = []
    selected = groups[0] if groups else None
    if selected is None:
        reasons.append("no_supported_core_proposal")
    else:
        if selected["touches_border"]:
            reasons.append("selected_support_touches_export_border")
        if selected["axis_ratio"] > config.max_axis_ratio:
            reasons.append("selected_support_elongated_or_fragmented")
        if selected["longest_flat_edge_perimeter_fraction"] > config.max_flat_edge_perimeter_fraction:
            reasons.append("selected_polygon_has_long_flat_edge")
        if selected["area"] > means.size * config.max_envelope_area_fraction:
            reasons.append("selected_envelope_excessive_export_fraction")
        if selected["center_distance"] > config.max_center_distance_fraction:
            reasons.append("target_prior_far_from_supported_core")
        if len(groups) > 1 and groups[1]["ranking_score"] - selected["ranking_score"] < config.ambiguity_margin:
            reasons.append("multiple_similarly_ranked_core_proposals")
    if omitted:
        reasons.append("object_proposal_budget_truncated")
    state = "unresolved" if reasons else "unreviewed"
    labels = np.zeros(means.shape, np.uint8)
    objects = []
    neighbour_guard = np.zeros(means.shape, np.uint8)
    for index, group in enumerate(groups):
        hull_mask = np.zeros(means.shape, np.uint8)
        cv2.fillConvexPoly(hull_mask, group["hull"], 1)
        if index:
            neighbour_guard |= cv2.dilate(hull_mask, _kernel(round(span * config.neighbour_guard_fraction)))
        objects.append({"proposal_id": f"core-{index:03d}",
                        "bbox_level0_xyxy": _native_box(group["bbox"], step, native.shape),
                        "polygon_level0_xy": [[min(w - 1, int(x) * step + (step - 1) / 2),
                                                min(h - 1, int(y) * step + (step - 1) / 2)] for x, y in group["hull"]],
                        "fragment_component_ids": group["members"], "axis_ratio": group["axis_ratio"],
                        "longest_flat_edge_perimeter_fraction": group["longest_flat_edge_perimeter_fraction"],
                        "support_touches_border": group["touches_border"],
                        "normalized_center_distance": group["center_distance"],
                        "ranking_score_not_probability": group["ranking_score"], "identity_reviewed": False})
    if selected is not None:
        envelope = np.zeros(means.shape, np.uint8)
        cv2.fillConvexPoly(envelope, selected["hull"], 1)
        if config.envelope_padding_cells:
            envelope = cv2.dilate(envelope, _kernel(config.envelope_padding_cells))
        overlap = int(np.count_nonzero(envelope & neighbour_guard))
        if overlap > max(1, int(envelope.sum()) * 0.01):
            reasons.append("core_envelope_overlaps_neighbour_guard")
            state = "unresolved"
        ring = cv2.dilate(envelope, _kernel(round(span * config.background_ring_fraction))).astype(bool)
        ring &= ~envelope.astype(bool) & ~neighbour_guard.astype(bool) & ~support
        labels[neighbour_guard.astype(bool)] = LABELS["neighbour_guard"]
        labels[ring] = LABELS["nearby_background"]
        labels[envelope.astype(bool)] = LABELS["core_dim_or_space"]
        labels[envelope.astype(bool) & support] = LABELS["tissue_support"]
    areas = label_counts_in_box(labels, step, native.shape, [0, 0, w, h])
    result = {"schema": "fluoinspect.core-proposals.v1", "engine_id": ENGINE_ID,
              "configuration": config.to_dict(), "source_shape_yx": [h, w],
              "label_shape_yx": list(labels.shape), "native_pixels_per_label_cell": step,
              "label_mapping": "Label cell XY covers [X*step,Y*step,(X+1)*step,(Y+1)*step), clipped to native source.",
              "label_codes": LABELS, "label_areas_native_px": areas,
              "target_point_native_xy": target,
              "selection_basis": "caller_supplied_unreviewed_point" if target_point_native_xy is not None else "export_center_prior_hypothesis",
              "target_identity_status": state, "unresolved_reasons": reasons,
              "selected_proposal_id": "core-000" if selected is not None else None,
              "proposals": objects, "omitted_object_proposals": omitted,
              "core_envelope_bbox_level0_xyxy": _label_bbox((labels == 1) | (labels == 2), step, native.shape),
              "background_bbox_level0_xyxy": _label_bbox(labels == 3, step, native.shape),
              "internal_dark_pixels_retained": True, "source_pixels_modified": False,
              "core_boundary_reviewed": False, "segmentation_accuracy_validated": False,
              "analysis_use": "provisional_scope_comparison_only" if state == "unresolved" else "unreviewed_analysis_scope",
              "limits": ["A central-position prior and convex hull can choose the wrong target or include empty space.",
                         "Coarse tissue support is intensity-derived and may exclude dim tissue; it never gates artifact observation.",
                         "Neighbour exclusion and grouping are provisional; uncertain cases require context/identity review.",
                         "Hull filling affects region labels only; missing image data cannot be restored."],
              "quality_decision": None}
    return result, labels


def measure_region_intensity(native, labels, step):
    """Original-pixel statistics within the exact recorded coarse-label footprints."""
    names = {"core_envelope": (1, 2), "tissue_support": (2,), "nearby_background": (3,)}
    if (not isinstance(native, np.ndarray) or isinstance(native, np.ma.MaskedArray)
            or native.ndim != 2 or native.dtype.kind != "u" or native.dtype.itemsize != 2):
        raise ValueError("Expected original uint16 YX values")
    label_counts_in_box(labels, step, native.shape, [0, 0, native.shape[1], native.shape[0]])
    totals = {name: {"pixels": 0, "sum": 0, "zero_pixels": 0} for name in names}
    xs = np.arange(native.shape[1]) // step
    for y0 in range(0, native.shape[0], 64):
        y1 = min(native.shape[0], y0 + 64)
        scope = labels[(np.arange(y0, y1) // step)[:, None], xs[None, :]]
        raw = native[y0:y1]
        for name, codes in names.items():
            values = raw[np.isin(scope, codes)]
            row = totals[name]
            row["pixels"] += int(values.size)
            row["sum"] += int(values.sum(dtype=np.uint64))
            row["zero_pixels"] += int(np.count_nonzero(values == 0))
    for row in totals.values():
        row["mean_native_units"] = row["sum"] / row["pixels"] if row["pixels"] else None
        row["zero_fraction"] = row["zero_pixels"] / row["pixels"] if row["pixels"] else None
        row["status"] = "measured_in_provisional_region" if row["pixels"] else "unassessed_empty_region"
    return totals
