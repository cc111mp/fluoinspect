"""Experimental local dark-region and axial intensity-step observations.

Every scheduled tile is evaluated, including unflagged/background areas. Mean
pooled proposals are checked against original uint16 rectangles. Natural tissue
spaces and boundaries can produce these observations: neither cause nor QC
acceptance is inferred. Evaluated footprints are not human/model inspection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import cv2
import numpy as np

from ..investigation.coverage import _anchors, _union_area, _validate_shape
from ..investigation.session import validate_box
from ..measurements.brightness import _block_mean

ENGINE_ID = "local_artifact_observations_v1"


@dataclass(frozen=True)
class LocalConfig:
    tile_size: int = 1024
    stride: int = 768
    max_tiles: int = 512
    proposal_max_dimension: int = 256
    widths_native_px: tuple[int, ...] = (16, 64, 256)
    min_log_contrast: float = 0.7
    min_boundary_log_contrast: float = 0.35
    min_bright_mean: float = 32.0
    min_step_support_fraction: float = 0.65
    min_step_length_native_px: int = 256
    max_candidates_per_tile: int = 8

    def __post_init__(self):
        for name, low, high in (("tile_size", 64, 2048), ("stride", 1, self.tile_size),
                                ("max_tiles", 0, 4096), ("proposal_max_dimension", 64, 512),
                                ("min_step_length_native_px", 16, 2048),
                                ("max_candidates_per_tile", 1, 32)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        for name, low, high in (("min_log_contrast", 0, 16),
                                ("min_boundary_log_contrast", 0, 16),
                                ("min_bright_mean", 0, 65535),
                                ("min_step_support_fraction", 0, 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"Invalid {name}")
        if (not isinstance(self.widths_native_px, tuple) or not 1 <= len(self.widths_native_px) <= 8
                or any(type(v) is not int or not 4 <= v <= 512 for v in self.widths_native_px)
                or len(set(self.widths_native_px)) != len(self.widths_native_px)):
            raise ValueError("Invalid native analysis widths")

    def to_dict(self):
        return asdict(self)


def tile_plan(shape_yx, config, target_box=None):
    """Independent detector budget; view-export budgets do not limit this scan."""
    _validate_shape(shape_yx)
    target = validate_box(target_box if target_box is not None
                          else [0, 0, shape_yx[1], shape_yx[0]], shape_yx)
    x0, y0, x1, y1 = target
    nx, last_x = _anchors(x1 - x0, config.tile_size, config.stride)
    ny, last_y = _anchors(y1 - y0, config.tile_size, config.stride)
    total = nx * ny
    count = min(config.max_tiles, total)
    indices = range(total) if count == total else [i * total // count for i in range(count)]
    boxes = []
    for index in indices:
        iy, ix = divmod(index, nx)
        x, y = x0 + min(ix * config.stride, last_x), y0 + min(iy * config.stride, last_y)
        boxes.append((index, [x, y, min(x + config.tile_size, x1), min(y + config.tile_size, y1)]))
    return target, total, boxes


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _stats(values):
    return {"mean": float(values.mean()), "zero_fraction": float(np.count_nonzero(values == 0) / values.size),
            "pixels": int(values.size)}


def _dark_proposals(raw, means, step, config):
    height, width = means.shape
    proposals, omissions = [], 0
    unresolved = []
    for requested in config.widths_native_px:
        cells = max(1, round(requested / step))
        cells += (cells % 2 == 0)
        outer = 3 * cells
        margin = outer // 2
        if outer > min(height, width):
            unresolved.append(requested)
            continue
        inner_mean = cv2.boxFilter(means, -1, (cells, cells), borderType=cv2.BORDER_REFLECT)
        outer_mean = cv2.boxFilter(means, -1, (outer, outer), borderType=cv2.BORDER_REFLECT)
        ring_mean = np.maximum(0, (outer_mean * outer ** 2 - inner_mean * cells ** 2)
                               / (outer ** 2 - cells ** 2))
        contrast = np.log1p(ring_mean) - np.log1p(inner_mean)
        valid = (contrast >= config.min_log_contrast) & (ring_mean >= config.min_bright_mean)
        valid[:margin] = valid[-margin:] = False
        valid[:, :margin] = valid[:, -margin:] = False
        count, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), connectivity=8)
        ranked = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
        omissions += max(0, len(ranked) - config.max_candidates_per_tile)
        for label in ranked[:config.max_candidates_per_tile]:
            yy, xx = np.where(labels == label)
            index = int(np.argmax(contrast[yy, xx]))
            row, column = int(yy[index]), int(xx[index])
            half = cells // 2
            inner = [(column - half) * step, (row - half) * step,
                     min(raw.shape[1], (column + half + 1) * step),
                     min(raw.shape[0], (row + half + 1) * step)]
            context = [(column - margin) * step, (row - margin) * step,
                       min(raw.shape[1], (column + margin + 1) * step),
                       min(raw.shape[0], (row + margin + 1) * step)]
            x0, y0, x1, y1 = inner
            cx0, cy0, cx1, cy1 = context
            inside = raw[y0:y1, x0:x1]
            around = raw[cy0:cy1, cx0:cx1]
            ring_pixels = around.size - inside.size
            ring_native_mean = max(0.0, (float(around.sum(dtype=np.float64))
                                        - float(inside.sum(dtype=np.float64))) / ring_pixels)
            native_contrast = math.log1p(ring_native_mean) - math.log1p(float(inside.mean()))
            if native_contrast < config.min_log_contrast or ring_native_mean < config.min_bright_mean:
                continue
            proposals.append({"kind": "local_dark_region", "bbox": inner, "context_bbox": context,
                              "analysis_width_native_px": requested,
                              "proposal_component_cells": int(stats[label, cv2.CC_STAT_AREA]),
                              "native_log_contrast": native_contrast,
                              "center": _stats(inside), "surround_mean": ring_native_mean,
                              "surround_pixels": int(ring_pixels),
                              "cause": "unresolved_tissue_space_or_signal_loss"})
    return proposals, omissions, unresolved


def _step_proposals(raw, means, step, config):
    proposals, omissions, unresolved = [], 0, []
    for axis in ("x", "y"):
        reduced = means if axis == "x" else means.T
        original = raw if axis == "x" else raw.T
        for requested in config.widths_native_px:
            cells = max(1, round(requested / step))
            if reduced.shape[1] <= 2 * cells or original.shape[0] < config.min_step_length_native_px:
                unresolved.append({"axis": axis, "width_native_px": requested})
                continue
            positions = np.arange(cells, reduced.shape[1] - cells)
            sums = np.pad(np.cumsum(reduced, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
            left = (sums[:, positions] - sums[:, positions - cells]) / cells
            right = (sums[:, positions + cells] - sums[:, positions]) / cells
            difference = np.log1p(right) - np.log1p(left)
            adjacent = np.log1p(reduced[:, positions]) - np.log1p(reduced[:, positions - 1])
            median = np.median(difference, axis=0)
            direction = np.where(median >= 0, 1, -1)
            support = ((difference * direction >= config.min_log_contrast)
                       & (adjacent * direction >= config.min_boundary_log_contrast)
                       & (np.maximum(left, right) >= config.min_bright_mean))
            fractions = support.mean(axis=0)
            eligible = fractions >= config.min_step_support_fraction
            peaks = [start + int(np.argmax(fractions[start:end])) for start, end in _runs(eligible)]
            peaks.sort(key=lambda i: float(fractions[i] * abs(median[i])), reverse=True)
            omissions += max(0, len(peaks) - config.max_candidates_per_tile)
            for index in peaks[:config.max_candidates_per_tile]:
                segments = _runs(support[:, index])
                if not segments:
                    continue
                start, end = max(segments, key=lambda pair: pair[1] - pair[0])
                a0, a1 = int(start * step), min(original.shape[0], int(end * step))
                if a1 - a0 < config.min_step_length_native_px:
                    continue
                boundary = int(positions[index] * step)
                flank = min(requested, boundary, original.shape[1] - boundary)
                native_left = original[a0:a1, boundary - flank:boundary]
                native_right = original[a0:a1, boundary:boundary + flank]
                d = np.log1p(native_right.mean(axis=1)) - np.log1p(native_left.mean(axis=1))
                near = min(4, flank)
                boundary_d = (np.log1p(native_right[:, :near].mean(axis=1))
                              - np.log1p(native_left[:, -near:].mean(axis=1)))
                sign = int(direction[index])
                verified = ((d * sign >= config.min_log_contrast)
                            & (boundary_d * sign >= config.min_boundary_log_contrast)
                            & (np.maximum(native_left.mean(axis=1), native_right.mean(axis=1))
                               >= config.min_bright_mean))
                fraction = float(verified.mean())
                if fraction < config.min_step_support_fraction:
                    continue
                box = [boundary - flank, a0, boundary + flank, a1]
                if axis == "y":
                    box = [box[1], box[0], box[3], box[2]]
                proposals.append({"kind": "axial_intensity_step", "bbox": box, "context_bbox": box,
                                  "axis": axis, "boundary_native_px": boundary,
                                  "analysis_width_native_px": requested,
                                  "native_log_contrast": float(np.median(d) * sign),
                                  "native_boundary_log_contrast": float(np.median(boundary_d) * sign),
                                  "native_support_fraction": fraction,
                                  "native_transects_checked": int(a1 - a0),
                                  "left_or_top": _stats(native_left), "right_or_bottom": _stats(native_right),
                                  "cause": "unresolved_tissue_boundary_or_technical_step"})
    return proposals, omissions, unresolved


def measure_local_artifacts(native, *, config=None, target_box=None):
    """Systematically measure scheduled tiles, without accepting negative results."""
    config = LocalConfig() if config is None else config
    if not isinstance(config, LocalConfig):
        raise ValueError("A LocalConfig is required")
    if (not isinstance(native, np.ndarray) or native.ndim != 2
            or native.dtype.kind != "u" or native.dtype.itemsize != 2):
        raise ValueError("Expected native uint16 YX pixels")
    target, total, boxes = tile_plan(native.shape, config, target_box)
    tiles, candidates = [], []
    proposals_omitted = 0
    for grid_index, box in boxes:
        x0, y0, x1, y1 = box
        raw = native[y0:y1, x0:x1]
        step = max(1, math.ceil(max(raw.shape) / config.proposal_max_dimension))
        means = _block_mean(raw, step)
        dark, dark_omitted, dark_unresolved = _dark_proposals(raw, means, step, config)
        edges, edge_omitted, edge_unresolved = _step_proposals(raw, means, step, config)
        proposed = sorted(dark + edges, key=lambda v: v["native_log_contrast"], reverse=True)
        omitted = dark_omitted + edge_omitted + max(0, len(proposed) - config.max_candidates_per_tile)
        proposals_omitted += omitted
        tile_id = f"tile-{grid_index:06d}"
        selected = proposed[:config.max_candidates_per_tile]
        for candidate in selected:
            for key in ("bbox", "context_bbox"):
                bx0, by0, bx1, by1 = candidate.pop(key)
                candidate[key + "_level0_xyxy"] = [bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]
            if "boundary_native_px" in candidate:
                candidate["boundary_native_px"] += x0 if candidate["axis"] == "x" else y0
            candidate.update(candidate_id=f"local-{len(candidates):06d}", tile_id=tile_id,
                             status="native_rectangle_measurements_verified", native_sampling_method="original_pixels",
                             artifact_cause_verified=False, quality_decision=None)
            candidates.append(candidate)
        tiles.append({"tile_id": tile_id, "bbox_level0_xyxy": box,
                      "status": "detector_evaluated", "proposal_native_pixels_per_cell": step,
                      "candidate_count": len(selected), "proposals_omitted": omitted,
                      "dark_widths_unresolved_native_px": dark_unresolved,
                      "step_scales_unresolved": edge_unresolved,
                      "model_inspection_logged": False, "human_inspection_logged": False})
    area = _union_area([tile["bbox_level0_xyxy"] for tile in tiles])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    return {"schema": "fluoinspect.local-artifacts.v1", "engine_id": ENGINE_ID,
            "configuration": config.to_dict(), "source_shape_yx": list(native.shape),
            "target_region_level0_xyxy": target, "region_identity_reviewed": False,
            "assessment": "experimental_measurements_only", "artifact_accuracy_validated": False,
            "source_pixels_modified": False, "dark_pixels_retained": True,
            "candidates": candidates, "candidate_count": len(candidates), "tiles": tiles,
            "coverage": {"total_grid_tiles": total, "evaluated_tiles": len(tiles),
                         "omitted_grid_tiles": total - len(tiles), "tile_budget_truncated": len(tiles) < total,
                         "target_area_px": target_area, "detector_evaluated_area_px": area,
                         "detector_evaluated_area_fraction": area / target_area,
                         "complete_scheduled_grid_evaluated": len(tiles) == total,
                         "candidate_proposals_omitted": proposals_omitted,
                         "detector_full_resolution_exhaustive": False,
                         "model_inspected_area_fraction": None, "human_inspected_area_fraction": None},
            "limits": ["Candidate rectangles can overlap across tiles/scales; counts are not unique artifacts.",
                       "Mean-pooled proposals may miss thin, weak, broad or oblique defects; raw verification checks proposed rectangles only.",
                       "Step directions are horizontal/vertical; dark-region contrasts do not identify cause.",
                       "Natural tissue spaces/boundaries and neighbouring cores can produce candidates.",
                       "Brightness and contrast thresholds are provisional native-unit settings, not assay-calibrated.",
                       "Detector-evaluated footprints do not establish exhaustive native-detail, human or model inspection."],
            "quality_decision": None}


def validate_local_evidence(result, shape_yx, methods):
    """Check scope, recorded footprints and interpretation when binding a record."""
    if (not isinstance(result, dict) or result.get("schema") != "fluoinspect.local-artifacts.v1"
            or result.get("engine_id") != ENGINE_ID or methods.get("engine_id") != ENGINE_ID
            or result.get("source_shape_yx") != list(shape_yx)
            or result.get("assessment") != "experimental_measurements_only"
            or result.get("artifact_accuracy_validated") is not False
            or result.get("source_pixels_modified") is not False
            or result.get("dark_pixels_retained") is not True
            or result.get("region_identity_reviewed") is not False
            or result.get("quality_decision") is not None
            or result.get("configuration") != methods.get("configuration")):
        raise ValueError("Local artifact scope or interpretation disagrees")
    options = dict(result["configuration"])
    options["widths_native_px"] = tuple(options["widths_native_px"])
    config = LocalConfig(**options)
    if json.dumps(config.to_dict(), sort_keys=True) != json.dumps(result["configuration"], sort_keys=True):
        raise ValueError("Local configuration disagrees")
    target, total, planned = tile_plan(shape_yx, config, result.get("target_region_level0_xyxy"))
    tiles = result.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != len(planned):
        raise ValueError("Local tile ledger disagrees")
    by_id = {}
    for tile, (index, box) in zip(tiles, planned):
        if (tile.get("tile_id") != f"tile-{index:06d}" or tile.get("bbox_level0_xyxy") != box
                or tile.get("status") != "detector_evaluated"
                or tile.get("model_inspection_logged") is not False
                or tile.get("human_inspection_logged") is not False
                or type(tile.get("proposals_omitted")) is not int or tile["proposals_omitted"] < 0
                or type(tile.get("candidate_count")) is not int
                or not 0 <= tile["candidate_count"] <= config.max_candidates_per_tile
                or tile.get("proposal_native_pixels_per_cell") != max(1, math.ceil(max(box[2] - box[0], box[3] - box[1]) / config.proposal_max_dimension))):
            raise ValueError("Local tile geometry or status disagrees")
        by_id[tile["tile_id"]] = tile
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or result.get("candidate_count") != len(candidates):
        raise ValueError("Local candidate ledger disagrees")
    counts = dict.fromkeys(by_id, 0)
    for index, candidate in enumerate(candidates):
        if (candidate.get("candidate_id") != f"local-{index:06d}"
                or candidate.get("tile_id") not in by_id
                or candidate.get("kind") not in {"local_dark_region", "axial_intensity_step"}
                or candidate.get("status") != "native_rectangle_measurements_verified"
                or candidate.get("native_sampling_method") != "original_pixels"
                or candidate.get("artifact_cause_verified") is not False
                or candidate.get("quality_decision") is not None):
            raise ValueError("Invalid local candidate scope")
        tile_box = by_id[candidate["tile_id"]]["bbox_level0_xyxy"]
        for key in ("bbox_level0_xyxy", "context_bbox_level0_xyxy"):
            box = validate_box(candidate.get(key), shape_yx)
            if box[0] < tile_box[0] or box[1] < tile_box[1] or box[2] > tile_box[2] or box[3] > tile_box[3]:
                raise ValueError("Local candidate lies outside its evaluated tile")
        counts[candidate["tile_id"]] += 1
    if any(tile["candidate_count"] != counts[key] for key, tile in by_id.items()):
        raise ValueError("Tile candidate count disagrees")
    area = _union_area([box for _, box in planned])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    expected = {"total_grid_tiles": total, "evaluated_tiles": len(planned),
                "omitted_grid_tiles": total - len(planned), "tile_budget_truncated": len(planned) < total,
                "target_area_px": target_area, "detector_evaluated_area_px": area,
                "detector_evaluated_area_fraction": area / target_area,
                "complete_scheduled_grid_evaluated": len(planned) == total,
                "candidate_proposals_omitted": sum(tile["proposals_omitted"] for tile in tiles),
                "detector_full_resolution_exhaustive": False,
                "model_inspected_area_fraction": None, "human_inspected_area_fraction": None}
    if result.get("coverage") != expected:
        raise ValueError("Local evaluated coverage disagrees")
    return result
