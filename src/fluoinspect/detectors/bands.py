"""Experimental bright–dark–bright band measurements on original uint16 pixels.

Dark pixels are retained. Oriented overview proposals are refined and checked
with sampled original-pixel transects. Tissue channels, folds and background
can produce the same observable pattern; no artifact cause or QC verdict follows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from ..io.native import MAX_NATIVE_PIXELS
from ..measurements.brightness import _block_mean

ENGINE_ID = "alternating_bands_v1"


@dataclass(frozen=True)
class BandConfig:
    region_xyxy: tuple[int, int, int, int] | None = None
    max_overview_dimension: int = 1024
    widths_native_px: tuple[int, ...] = (4, 16, 64, 256, 768)
    angles_degrees: tuple[float, ...] = (0, -1, 1, -3, 3, -15, 15, -45, 45, 90)
    min_length_native_px: int = 512
    min_depth_log1p: float = 0.12
    min_support_fraction: float = 0.6
    min_bin_support_fraction: float = 0.4
    max_gap_native_px: int = 64
    max_proposals: int = 24
    max_native_samples: int = 512
    max_normal_samples: int = 2048
    max_angle_refinement_degrees: float = 5
    edge_drop_fraction: float = 0.5
    max_edge_residual_width_fraction: float = 0.15
    max_width_dispersion_fraction: float = 0.25

    def __post_init__(self):
        for name, low, high in (
            ("max_overview_dimension", 128, 2048),
            ("min_length_native_px", 32, 65536),
            ("max_gap_native_px", 0, 512),
            ("max_proposals", 1, 64),
            ("max_native_samples", 32, 2048),
            ("max_normal_samples", 128, 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        for name, low, high in (
            ("min_depth_log1p", 0, 16),
            ("min_support_fraction", 0, 1),
            ("min_bin_support_fraction", 0, 1),
            ("max_angle_refinement_degrees", 0, 20),
            ("edge_drop_fraction", 0, 1),
            ("max_edge_residual_width_fraction", 0, 1),
            ("max_width_dispersion_fraction", 0, 1),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low < value <= high
            ):
                raise ValueError(f"Invalid {name}")
        if self.min_bin_support_fraction > self.min_support_fraction:
            raise ValueError("Bin support cannot exceed total required support")
        if (
            not isinstance(self.widths_native_px, tuple)
            or not 1 <= len(self.widths_native_px) <= 8
            or any(
                type(v) is not int or not 2 <= v <= 2048 for v in self.widths_native_px
            )
            or len(set(self.widths_native_px)) != len(self.widths_native_px)
        ):
            raise ValueError("Invalid native band analysis widths")
        if (
            not isinstance(self.angles_degrees, tuple)
            or not 1 <= len(self.angles_degrees) <= 24
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not -90 <= v < 180
                for v in self.angles_degrees
            )
            or len({v % 180 for v in self.angles_degrees}) != len(self.angles_degrees)
        ):
            raise ValueError("Invalid band orientations")
        if self.region_xyxy is not None and (
            not isinstance(self.region_xyxy, tuple)
            or len(self.region_xyxy) != 4
            or any(type(v) is not int for v in self.region_xyxy)
            or not 0 <= self.region_xyxy[0] < self.region_xyxy[2]
            or not 0 <= self.region_xyxy[1] < self.region_xyxy[3]
        ):
            raise ValueError("Invalid source region")

    def to_dict(self):
        return asdict(self)


def _angle_distance(a, b):
    return abs((a - b + 90) % 180 - 90)


def _rotate(levels, angle):
    height, width = levels.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
    corners = (
        np.asarray(
            [
                [0, 0, 1],
                [width - 1, 0, 1],
                [0, height - 1, 1],
                [width - 1, height - 1, 1],
            ]
        )
        @ matrix.T
    )
    low, high = corners.min(axis=0), corners.max(axis=0)
    matrix[:, 2] -= low
    shape = tuple((np.ceil(high - low) + 1).astype(int))
    values = cv2.warpAffine(
        levels,
        matrix,
        shape,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    weight = cv2.warpAffine(
        np.ones_like(levels),
        matrix,
        shape,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return values, weight, cv2.invertAffineTransform(matrix)


def _source_point(point, inverse, shape, step, box):
    column, row = inverse @ np.r_[point, 1.0]
    height, width = shape
    xs = np.arange(0, width, step)
    ys = np.arange(0, height, step)
    x_centers = xs + (np.minimum(step, width - xs) - 1) / 2
    y_centers = ys + (np.minimum(step, height - ys) - 1) / 2
    return np.asarray(
        [
            box[0] + np.interp(column, np.arange(len(xs)), x_centers),
            box[1] + np.interp(row, np.arange(len(ys)), y_centers),
        ],
        dtype=float,
    )


def _overlap(a, b):
    t = np.asarray(a["tangent"])
    av = sorted(np.asarray(a["endpoints"]) @ t)
    bv = sorted(np.asarray(b["endpoints"]) @ t)
    return max(0, min(av[1], bv[1]) - max(av[0], bv[0])) / max(
        1, min(av[1] - av[0], bv[1] - bv[0])
    )


def _same_band(a, b):
    if (
        _angle_distance(a["angle_degrees"], b["angle_degrees"]) > 3
        or _overlap(a, b) < 0.5
    ):
        return False
    tangent = np.asarray(a["tangent"])
    normal = np.asarray([-tangent[1], tangent[0]])
    ca, cb = np.mean(a["endpoints"], axis=0), np.mean(b["endpoints"], axis=0)
    return abs(float((cb - ca) @ normal)) <= 0.6 * max(
        a["analysis_width_native_px"], b["analysis_width_native_px"]
    )


def _propose(levels, shape, step, box, config):
    proposals = []
    unsupported = []
    per_scale_truncated = False
    minimum = max(4, math.ceil(config.min_length_native_px / step))
    gap = max(1, math.ceil(config.max_gap_native_px / step))
    resolved = [w for w in config.widths_native_px if w / step >= 2]
    unsupported.extend(
        {"width_native_px": w, "reason": "less_than_two_overview_samples"}
        for w in config.widths_native_px
        if w not in resolved
    )
    assessed_widths = set()
    for angle in config.angles_degrees:
        rotated, weight, inverse = _rotate(levels, angle)
        if rotated.shape[1] < minimum:
            continue
        for requested in resolved:
            width = max(2, int(round(requested / step)))
            if rotated.shape[0] < 3 * width + 2:
                continue
            values = cv2.boxFilter(
                rotated, -1, (1, width), normalize=True, borderType=cv2.BORDER_CONSTANT
            )
            coverage = cv2.boxFilter(
                weight, -1, (1, width), normalize=True, borderType=cv2.BORDER_CONSTANT
            )
            left, right = (
                np.roll(values, width, axis=0),
                np.roll(values, -width, axis=0),
            )
            valid = (
                (coverage >= 0.999)
                & (np.roll(coverage, width, axis=0) >= 0.999)
                & (np.roll(coverage, -width, axis=0) >= 0.999)
            )
            valid[: 2 * width] = False
            valid[-2 * width :] = False
            if not valid.any():
                continue
            assessed_widths.add(requested)
            depth = np.minimum(left, right) - values
            hits = (depth >= config.min_depth_log1p) & valid
            bridged = (
                cv2.morphologyEx(
                    hits.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    np.ones((1, 2 * gap + 1), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                ).astype(bool)
                & valid
            )
            local = []
            for y in np.flatnonzero(bridged.sum(axis=1) >= minimum):
                changes = np.flatnonzero(
                    np.diff(np.r_[False, bridged[y], False].astype(np.int8))
                )
                for x0, x1 in zip(changes[::2], changes[1::2]):
                    if x1 - x0 < minimum:
                        continue
                    support = hits[y, x0:x1]
                    if (
                        support.mean() < config.min_support_fraction
                        or min(chunk.mean() for chunk in np.array_split(support, 4))
                        < config.min_bin_support_fraction
                    ):
                        continue
                    score = float(
                        np.median(depth[y, x0:x1][support])
                        * support.mean()
                        * math.sqrt(x1 - x0)
                    )
                    local.append((score, int(y), int(x0), int(x1 - 1)))
            retained = []
            for score, y, x0, x1 in sorted(local, reverse=True):
                if any(
                    abs(y - py) <= max(1, width // 2)
                    and min(x1, px1) - max(x0, px0) > 0.5 * min(x1 - x0, px1 - px0)
                    for _, py, px0, px1 in retained
                ):
                    continue
                retained.append((score, y, x0, x1))
                if len(retained) == 5:
                    per_scale_truncated = True
                    retained.pop()
                    break
            for score, y, x0, x1 in retained:
                start = _source_point([x0, y], inverse, shape, step, box)
                end = _source_point([x1, y], inverse, shape, step, box)
                delta = end - start
                length = float(np.linalg.norm(delta))
                if length < config.min_length_native_px:
                    continue
                tangent = delta / length
                proposals.append(
                    {
                        "endpoints": [start.tolist(), end.tolist()],
                        "tangent": tangent.tolist(),
                        "angle_degrees": float(
                            math.degrees(math.atan2(tangent[1], tangent[0])) % 180
                        ),
                        "analysis_width_native_px": requested,
                        "proposal_score": score,
                    }
                )
    proposals.sort(key=lambda p: -p["proposal_score"])
    unique = []
    for proposal in proposals:
        if not any(_same_band(proposal, p) for p in unique):
            unique.append(proposal)
    unsupported.extend(
        {
            "width_native_px": w,
            "reason": "insufficient_overview_flank_context_or_length",
        }
        for w in resolved
        if w not in assessed_widths
    )
    return (
        unique[: config.max_proposals],
        len(unique),
        sorted(assessed_widths),
        unsupported,
        per_scale_truncated,
    )


def _clip_line(start, tangent, length, box, normal_margin):
    normal = np.asarray([-tangent[1], tangent[0]])
    lo = np.asarray(box[:2], float) + np.abs(normal) * normal_margin + 1
    hi = np.asarray(box[2:], float) - np.abs(normal) * normal_margin - 2
    a, z = 0.0, length
    for i in range(2):
        if abs(tangent[i]) < 1e-10:
            if not lo[i] <= start[i] <= hi[i]:
                return None
        else:
            first, last = sorted(
                [(lo[i] - start[i]) / tangent[i], (hi[i] - start[i]) / tangent[i]]
            )
            a, z = max(a, first), min(z, last)
    return (start + a * tangent, z - a) if z > a else None


def _transects(native, start, tangent, length, width, box, config):
    count = min(config.max_native_samples, max(32, math.ceil(length / 2)))
    distance = np.linspace(0, length, count)
    normal = np.asarray([-tangent[1], tangent[0]])
    stride = max(1, math.ceil((6 * width + 1) / config.max_normal_samples))
    offsets = np.arange(-3 * width, 3 * width + 1, stride, dtype=float)
    centers = start + distance[:, None] * tangent
    points = centers[:, None, :] + offsets[None, :, None] * normal
    xs, ys = np.rint(points[..., 0]).astype(int), np.rint(points[..., 1]).astype(int)
    valid = (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
    sampled = np.zeros(xs.shape, np.float64)
    sampled[valid] = native[ys[valid], xs[valid]]
    cells = max(1, round(width / stride))
    sums = np.c_[np.zeros(count), np.cumsum(sampled, axis=1)]
    counts = np.c_[np.zeros(count), np.cumsum(valid, axis=1)]
    means = (sums[:, cells:] - sums[:, :-cells]) / cells
    zero_sums = np.c_[np.zeros(count), np.cumsum(valid & (sampled == 0), axis=1)]
    zero_fraction = (zero_sums[:, cells:] - zero_sums[:, :-cells]) / cells
    square_sums = np.c_[np.zeros(count), np.cumsum(np.square(sampled), axis=1)]
    deviations = np.sqrt(
        np.maximum(
            0,
            (square_sums[:, cells:] - square_sums[:, :-cells]) / cells - means * means,
        )
    )
    complete = (counts[:, cells:] - counts[:, :-cells]) == cells
    center = means[:, cells:-cells]
    left, right = means[:, : -2 * cells], means[:, 2 * cells :]
    supported = (
        complete[:, cells:-cells] & complete[:, : -2 * cells] & complete[:, 2 * cells :]
    )
    positions = ((offsets[: len(means[0])] + offsets[cells - 1 :]) / 2)[cells:-cells]
    depth = np.minimum(np.log1p(left), np.log1p(right)) - np.log1p(center)
    return {
        "distance": distance,
        "positions": positions,
        "left": left,
        "center": center,
        "right": right,
        "depth": depth,
        "supported": supported,
        "stride": stride,
        "actual_width": cells * stride,
        "sampled_native_values": sampled,
        "sampled_native_valid": valid,
        "normal_offsets": offsets,
        "band_zero_fraction": zero_fraction[:, cells:-cells],
        "band_std": deviations[:, cells:-cells],
        "left_std": deviations[:, : -2 * cells],
        "right_std": deviations[:, 2 * cells :],
    }


def _dark_edges(transects, width, left, right, config):
    """Describe native low-signal run edges; these are not tissue identities."""
    values = transects["sampled_native_values"]
    offsets = transects["normal_offsets"]
    smooth_cells = max(1, round(max(2, width / 16) / transects["stride"]))
    sums = np.c_[np.zeros(len(values)), np.cumsum(values, axis=1)]
    counts = np.c_[
        np.zeros(len(values)), np.cumsum(transects["sampled_native_valid"], axis=1)
    ]
    smoothed = (sums[:, smooth_cells:] - sums[:, :-smooth_cells]) / smooth_cells
    complete = (counts[:, smooth_cells:] - counts[:, :-smooth_cells]) == smooth_cells
    positions = (offsets[: smoothed.shape[1]] + offsets[smooth_cells - 1 :]) / 2
    threshold = np.minimum(left, right) * (1 - config.edge_drop_fraction)
    mask = (
        (smoothed <= threshold[:, None])
        & complete
        & (np.abs(positions)[None] <= 2 * width)
    )
    starts, ends = np.full(len(values), np.nan), np.full(len(values), np.nan)
    for row in range(len(values)):
        changes = np.flatnonzero(
            np.diff(np.r_[False, mask[row], False].astype(np.int8))
        )
        runs = [
            (a, z)
            for a, z in zip(changes[::2], changes[1::2])
            if a > 0
            and z < len(positions)
            and np.min(np.abs(positions[a:z])) <= 0.35 * width
        ]
        if not runs:
            continue
        a, z = min(runs, key=lambda r: np.min(np.abs(positions[r[0] : r[1]])))
        if (
            a == 0
            or z == len(positions)
            or positions[a] <= -2 * width
            or positions[z - 1] >= 2 * width
        ):
            continue
        starts[row] = positions[a] - transects["stride"] / 2
        ends[row] = positions[z - 1] + transects["stride"] / 2
    eligible = np.isfinite(starts) & np.isfinite(ends)
    if eligible.sum() < 16:
        return {
            "status": "unassessed",
            "reason": "insufficient_bounded_low_signal_runs",
            "straight_edged_loss_pattern": False,
        }
    distance = transects["distance"][eligible]
    widths = ends[eligible] - starts[eligible]
    median_width = float(np.median(widths))
    residuals = []
    for values in (starts[eligible], ends[eligible]):
        slope, intercept = np.polyfit(distance, values, 1)
        residuals.append(
            float(np.percentile(np.abs(values - (slope * distance + intercept)), 90))
        )
    dispersion = float(np.median(np.abs(widths - median_width)) / max(median_width, 1))
    bins = [float(v.mean()) for v in np.array_split(eligible, 4)]
    straight = (
        eligible.mean() >= config.min_support_fraction
        and min(bins) >= config.min_bin_support_fraction
        and max(residuals)
        <= max(2, config.max_edge_residual_width_fraction * median_width)
        and dispersion <= config.max_width_dispersion_fraction
    )
    return {
        "status": "measured",
        "edge_drop_fraction": config.edge_drop_fraction,
        "bounded_dark_run_fraction": float(eligible.mean()),
        "bin_edge_support_fractions": bins,
        "median_thresholded_dark_run_width_native_px": median_width,
        "left_edge_p90_fit_residual_native_px": residuals[0],
        "right_edge_p90_fit_residual_native_px": residuals[1],
        "width_median_absolute_deviation_fraction": dispersion,
        "straight_edged_loss_pattern": bool(straight),
        "interpretation": "sampled low-signal runs with nearly straight stable edges; cause and quality unassessed",
    }


def _verify(native, proposal, box, config):
    start = np.asarray(proposal["endpoints"][0])
    tangent = np.asarray(proposal["tangent"])
    length = float(np.linalg.norm(np.asarray(proposal["endpoints"][1]) - start))
    width = proposal["analysis_width_native_px"]
    clipped = _clip_line(start, tangent, length, box, 3 * width)
    if clipped is None or clipped[1] < config.min_length_native_px:
        return {
            "status": "unassessed",
            "reason": "insufficient_original_pixel_flank_context",
            "proposal": proposal,
        }
    start, length = clipped
    traces = _transects(native, start, tangent, length, width, box, config)
    allowed = np.abs(traces["positions"]) <= 1.5 * width
    candidates = np.where(traces["supported"] & allowed[None], traces["depth"], -np.inf)
    best = candidates.argmax(axis=1)
    scores = candidates[np.arange(len(best)), best]
    selected = scores >= config.min_depth_log1p
    if selected.sum() < 16:
        return {
            "status": "not_verified",
            "reason": "insufficient_dark_band_transects",
            "proposal": proposal,
        }
    distance, offsets = traces["distance"], traces["positions"][best]
    keep = selected.copy()
    for _ in range(4):
        if keep.sum() < 16 or np.ptp(distance[keep]) < config.min_length_native_px / 2:
            return {
                "status": "not_verified",
                "reason": "insufficient_geometry_support",
                "proposal": proposal,
            }
        slope, intercept = np.polyfit(distance[keep], offsets[keep], 1)
        residual = offsets - (slope * distance + intercept)
        keep = selected & (np.abs(residual) <= max(2, 0.3 * width))
    if abs(math.degrees(math.atan(slope))) > config.max_angle_refinement_degrees:
        return {
            "status": "not_verified",
            "reason": "refinement_exceeds_orientation_scope",
            "proposal": proposal,
        }
    normal = np.asarray([-tangent[1], tangent[0]])
    refined_start = start + intercept * normal
    refined_tangent = tangent + slope * normal
    refined_tangent /= np.linalg.norm(refined_tangent)
    refined_length = length * math.sqrt(1 + slope * slope)
    clipped = _clip_line(refined_start, refined_tangent, refined_length, box, 3 * width)
    if clipped is None or clipped[1] < config.min_length_native_px:
        return {
            "status": "unassessed",
            "reason": "insufficient_refined_flank_context",
            "proposal": proposal,
        }
    refined_start, refined_length = clipped
    checked = _transects(
        native, refined_start, refined_tangent, refined_length, width, box, config
    )
    at_zero = int(np.argmin(np.abs(checked["positions"])))
    left, center, right = (
        checked[name][:, at_zero] for name in ("left", "center", "right")
    )
    depth = checked["depth"][:, at_zero]
    supported = checked["supported"][:, at_zero]
    dark = supported & (depth >= config.min_depth_log1p)
    bins = [float(v.mean()) for v in np.array_split(dark, 4)]
    if (
        dark.mean() < config.min_support_fraction
        or min(bins) < config.min_bin_support_fraction
    ):
        return {
            "status": "not_verified",
            "reason": "insufficient_continuous_native_support",
            "proposal": proposal,
            "native_support_fraction": float(dark.mean()),
            "bin_support_fractions": bins,
        }
    end = refined_start + refined_length * refined_tangent
    normal = np.asarray([-refined_tangent[1], refined_tangent[0]])
    corners = [refined_start + sign * 1.5 * width * normal for sign in (-1, 1)]
    corners += [end + sign * 1.5 * width * normal for sign in (1, -1)]
    extent = np.asarray(corners)
    context = [
        max(box[0], int(math.floor(extent[:, 0].min()))),
        max(box[1], int(math.floor(extent[:, 1].min()))),
        min(box[2], int(math.ceil(extent[:, 0].max())) + 1),
        min(box[3], int(math.ceil(extent[:, 1].max())) + 1),
    ]
    trace_indices = np.unique(np.linspace(0, len(depth) - 1, 12).astype(int))
    edges = _dark_edges(checked, width, left, right, config)
    return {
        "status": "native_verified_pattern",
        "candidate_kind": "dark_band_between_brighter_flanks",
        "endpoints": [refined_start.tolist(), end.tolist()],
        "tangent": refined_tangent.tolist(),
        "angle_degrees": float(
            math.degrees(math.atan2(refined_tangent[1], refined_tangent[0])) % 180
        ),
        "analysis_width_native_px": width,
        "actual_sampled_width_native_px": checked["actual_width"],
        "length_native_px": refined_length,
        "context_bbox_level0_xyxy": context,
        "native_sample_count": len(depth),
        "native_normal_sampling_step_px": checked["stride"],
        "native_support_fraction": float(dark.mean()),
        "bin_support_fractions": bins,
        "median_depth_log1p": float(np.median(depth[dark])),
        "median_band_mean_native_intensity": float(np.median(center[dark])),
        "median_left_flank_mean_native_intensity": float(np.median(left[dark])),
        "median_right_flank_mean_native_intensity": float(np.median(right[dark])),
        "median_relative_intensity_drop": float(
            np.median(
                1 - center[dark] / np.maximum(np.minimum(left[dark], right[dark]), 1)
            )
        ),
        "median_exact_zero_fraction_in_band_samples": float(
            np.median(checked["band_zero_fraction"][dark, at_zero])
        ),
        "median_band_to_flank_texture_std_ratio": float(
            np.median(
                checked["band_std"][dark, at_zero]
                / np.maximum(
                    np.minimum(
                        checked["left_std"][dark, at_zero],
                        checked["right_std"][dark, at_zero],
                    ),
                    1,
                )
            )
        ),
        "dark_edge_geometry": edges,
        "geometry_fit_residual_native_px": float(np.median(np.abs(residual[keep]))),
        "sample_trace": [
            {
                "distance_native_px": float(checked["distance"][i]),
                "left_mean": float(left[i]),
                "band_mean": float(center[i]),
                "right_mean": float(right[i]),
                "depth_log1p": float(depth[i]),
                "supports_pattern": bool(dark[i]),
            }
            for i in trace_indices
        ],
        "sample_trace_scope": "twelve or fewer audit examples, not the complete checked transects",
        "native_sampling_method": "nearest_original_pixels",
        "interpretation": "observed contrast geometry; artifact cause and acceptance unassessed",
    }


def _pairs(native, bands, box, config):
    pairs = []
    for i, first in enumerate(bands):
        tangent = np.asarray(first["tangent"])
        normal = np.asarray([-tangent[1], tangent[0]])
        a = np.asarray(first["endpoints"])
        for second in bands[i + 1 :]:
            if (
                _angle_distance(first["angle_degrees"], second["angle_degrees"]) > 3
                or _overlap(first, second) < 0.5
            ):
                continue
            b = np.asarray(second["endpoints"])
            spacing = abs(float((b.mean(axis=0) - a.mean(axis=0)) @ normal))
            widths = (
                first["analysis_width_native_px"] + second["analysis_width_native_px"]
            )
            if spacing <= widths:
                continue
            av, bv = sorted(a @ tangent), sorted(b @ tangent)
            low, high = max(av[0], bv[0]), min(av[1], bv[1])
            first_normals = np.interp([low, high], a @ tangent, a @ normal)
            second_normals = np.interp([low, high], b @ tangent, b @ normal)
            separation = second_normals - first_normals
            if (
                separation.prod() <= 0
                or np.abs(separation).min() <= widths
                or np.ptp(separation) > 0.25 * spacing
            ):
                continue
            midpoint_normal = float((a.mean(axis=0) + b.mean(axis=0)) @ normal / 2)
            middle = low * tangent + midpoint_normal * normal
            gap_width = max(
                2,
                min(
                    int(spacing / 3),
                    max(
                        first["analysis_width_native_px"],
                        second["analysis_width_native_px"],
                    ),
                ),
            )
            sample = _transects(
                native, middle, tangent, high - low, gap_width, box, config
            )
            at_zero = int(np.argmin(np.abs(sample["positions"])))
            middle_mean = sample["center"][:, at_zero]
            valid = sample["supported"][:, at_zero]
            dark_mean = max(
                first["median_band_mean_native_intensity"],
                second["median_band_mean_native_intensity"],
            )
            brighter = valid & (
                np.log1p(middle_mean) - math.log1p(dark_mean) >= config.min_depth_log1p
            )
            if brighter.mean() >= config.min_support_fraction:
                pairs.append(
                    {
                        "band_ids": [first["candidate_id"], second["candidate_id"]],
                        "spacing_native_px": spacing,
                        "overlapping_length_native_px": high - low,
                        "brighter_between_support_fraction": float(brighter.mean()),
                        "pattern": "brighter_flank_dark_band_brighter_region_dark_band_brighter_flank",
                        "interpretation": "parallel alternating intensity pattern; no periodicity or artifact cause established",
                    }
                )
    return pairs


def measure_alternating_bands(native, *, config=None):
    config = BandConfig() if config is None else config
    if not isinstance(config, BandConfig):
        raise ValueError("A BandConfig is required")
    if (
        not isinstance(native, np.ndarray)
        or isinstance(native, np.ma.MaskedArray)
        or native.ndim != 2
        or not native.size
        or native.dtype.kind != "u"
        or native.dtype.itemsize != 2
        or native.size > MAX_NATIVE_PIXELS
    ):
        raise ValueError("Expected bounded, nonempty uint16 YX pixels")
    box = (
        list(config.region_xyxy)
        if config.region_xyxy is not None
        else [0, 0, native.shape[1], native.shape[0]]
    )
    if box[2] > native.shape[1] or box[3] > native.shape[0]:
        raise ValueError("Band region exceeds the source image")
    region = native[box[1] : box[3], box[0] : box[2]]
    step = max(1, math.ceil(max(region.shape) / config.max_overview_dimension))
    overview = _block_mean(region, step)
    if overview.max() == 0:
        proposals, proposed_count, assessed_widths = [], 0, []
        per_scale_truncated = False
        unsupported = [
            {"width_native_px": w, "reason": "no_positive_signal"}
            for w in config.widths_native_px
        ]
    else:
        proposals, proposed_count, assessed_widths, unsupported, per_scale_truncated = (
            _propose(np.log1p(overview), region.shape, step, box, config)
        )
    records = [_verify(native, proposal, box, config) for proposal in proposals]
    verified = []
    for record in sorted(
        (r for r in records if r["status"] == "native_verified_pattern"),
        key=lambda r: -r["median_depth_log1p"] * r["native_support_fraction"],
    ):
        if not any(_same_band(record, p) for p in verified):
            record["candidate_id"] = f"band-{len(verified) + 1:03d}"
            verified.append(record)
    pairs = _pairs(native, verified, box, config)
    straight_ids = {
        b["candidate_id"]
        for b in verified
        if b["dark_edge_geometry"].get("straight_edged_loss_pattern") is True
    }
    return {
        "schema": "fluoinspect.alternating-bands.v1",
        "engine_id": ENGINE_ID,
        "configuration": config.to_dict(),
        "source_shape_yx": list(native.shape),
        "source_region_level0_xyxy": box,
        "region_identity_reviewed": False,
        "region_selection": "caller_supplied_rectangle_unreviewed"
        if config.region_xyxy is not None
        else "whole_export_unreviewed",
        "dark_pixels_retained": True,
        "overview_step_native_px": step,
        "overview_shape_yx": list(overview.shape),
        "assessed_analysis_widths_native_px": assessed_widths,
        "unassessed_widths": unsupported,
        "screen_status": "measured" if assessed_widths else "unassessed",
        "proposal_count_before_budget": proposed_count,
        "proposal_budget_truncated": proposed_count > config.max_proposals
        or per_scale_truncated,
        "per_orientation_width_proposal_limit": 4,
        "per_orientation_width_proposals_truncated": per_scale_truncated,
        "checked_proposal_count": len(records),
        "native_verified_bands": verified,
        "unverified_proposals": [
            r for r in records if r["status"] != "native_verified_pattern"
        ],
        "parallel_alternating_pairs": pairs,
        "straight_edged_alternating_pairs": [
            p for p in pairs if all(v in straight_ids for v in p["band_ids"])
        ],
        "assessment": "experimental_measurements_only",
        "artifact_accuracy_validated": False,
        "quality_decision": None,
        "source_pixels_modified": False,
        "caveats": [
            "ROI/core/tissue identity is unreviewed; a whole export may include neighbouring cores and background.",
            "Natural tissue channels, vessels, clefts, folds and background can produce this same contrast geometry.",
            "Native verification establishes sampled intensity geometry, not acquisition/correction cause or quality rejection.",
            "Only the recorded scales, proposal budget and tested orientations are inspected; no candidates is not image acceptance.",
            "Widths smaller than two overview samples require finer native-region investigation.",
            "Requested analysis width is a sampling scale, not an expert-validated physical band width.",
            "Straight low-signal edges, zero fractions and texture loss are additional observable descriptors, not calibrated artifact classes.",
            "A dark area touching the region edge may lack brighter-flank context and remain unassessed.",
        ],
    }
