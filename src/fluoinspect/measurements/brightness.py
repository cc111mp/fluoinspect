"""Experimental broad and periodic brightness measurements on original pixels.

These measures describe intensity variation. Tissue structure can produce similar
variation; no artifact cause, quality class, or acceptance probability is assigned.
The periodicity hypothesis must be supplied explicitly, in native pixel units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import cv2
import numpy as np

from ..detectors import axial
from ..io.native import MAX_NATIVE_PIXELS

ENGINE_ID = "brightness_patterns_v1"


@dataclass(frozen=True)
class BrightnessConfig:
    max_overview_dimension: int = 1024
    period_native_px: float | None = None
    period_basis: str = "hypothesis"
    foreground_method: str = "intensity_otsu"
    phase_bins: int = 24
    min_cycles: int = 3
    min_cycle_samples: int = 24
    min_profile_coverage: float = 0.15
    erosion_cells: int = 3
    trend_reduction: int = 16
    control_period_factors: tuple[float, ...] = (0.77, 0.87, 1.17, 1.31)
    broad_window_native_px: tuple[int, ...] = (32, 128, 512)

    def __post_init__(self):
        for name, low, high in (
            ("max_overview_dimension", 64, 2048), ("phase_bins", 8, 64),
            ("min_cycles", 3, 32), ("min_cycle_samples", 8, 512),
            ("erosion_cells", 0, 16), ("trend_reduction", 1, 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        if (type(self.min_profile_coverage) not in (int, float)
                or not math.isfinite(self.min_profile_coverage)
                or not 0 < self.min_profile_coverage <= 1):
            raise ValueError("Invalid profile coverage")
        if self.period_native_px is not None and (
            type(self.period_native_px) not in (int, float)
            or not math.isfinite(self.period_native_px) or self.period_native_px <= 0
        ):
            raise ValueError("Period must be finite and positive, or absent")
        if self.period_basis not in {"hypothesis", "acquisition_metadata"}:
            raise ValueError("Period basis must be a hypothesis or supplied metadata")
        if self.foreground_method not in {"intensity_otsu", "hysteresis", "region_envelope"}:
            raise ValueError("Unsupported provisional foreground method")
        if self.foreground_method == "region_envelope" and self.erosion_cells != 0:
            raise ValueError("Region-envelope measurements require explicit zero erosion")
        if (not isinstance(self.control_period_factors, tuple)
                or not 1 <= len(self.control_period_factors) <= 8
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or v <= 0 or v == 1 for v in self.control_period_factors)):
            raise ValueError("Invalid control periods")
        if (not isinstance(self.broad_window_native_px, tuple)
                or not 1 <= len(self.broad_window_native_px) <= 8
                or any(type(v) is not int or not 1 <= v <= 65536
                       for v in self.broad_window_native_px)):
            raise ValueError("Invalid broad windows")

    def to_dict(self):
        return asdict(self)


def _block_mean(values, step):
    height, width = values.shape
    xs = np.arange(0, width, step)
    counts = np.minimum(step, width - xs)
    result = np.empty((math.ceil(height / step), len(xs)), np.float32)
    for index, y0 in enumerate(range(0, height, step)):
        stripe = values[y0:min(height, y0 + step)].astype(np.float64)
        result[index] = np.add.reduceat(stripe, xs, axis=1).sum(axis=0) / (len(stripe) * counts)
    return result


def _profile(values, mask, axis, minimum):
    data, eligible = (values, mask) if axis == "x" else (values.T, mask.T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(np.where(eligible, data, np.nan), axis=0)
    valid = (eligible.mean(axis=0) >= minimum) & np.isfinite(profile)
    return np.where(valid, profile, 0).astype(np.float64), valid


def _intensity_region(levels):
    """Provisional nonzero-intensity foreground, not a reviewed tissue identity.

    Separate the low-intensity mode from brighter pixels with histogram Otsu;
    require separation from the lower side of that mode to limit dim halos.
    Original zero pixels remain excluded rather than being asserted as glass.
    """
    values = levels[::2, ::2].ravel()
    values = values[values > 0]
    if len(values) < 16 or np.ptp(values) < 1e-6:
        return np.zeros_like(levels, bool)
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.cumsum(hist)
    remaining = counts[-1] - counts
    sums = np.cumsum(hist * centers)
    left = sums / np.maximum(counts, 1)
    right = (sums[-1] - sums) / np.maximum(remaining, 1)
    split = float(centers[np.argmax(counts * remaining * np.square(left - right))])
    lower = values[values < split]
    if len(lower) < 100:
        lower = values[values <= np.percentile(values, 20)]
    hist, edges = np.histogram(lower, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    mode = float(centers[np.argmax(np.convolve(hist, np.ones(5) / 5, "same"))])
    below = lower[lower <= mode]
    spread = float(np.sqrt(np.mean(np.square(below - mode)))) if len(below) > 100 else 0.1
    high = float(np.percentile(values, 99.5))
    threshold = min(max(split, mode + 6 * spread), mode + 0.6 * (high - mode))
    mask = cv2.morphologyEx((levels > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 16
    return keep[labels]


def _broad_summary(profile, valid, axis, shape, step, config):
    count = int(valid.sum())
    if count < 2:
        return {"status": "unassessed", "reason": "insufficient_region_support", "supported_samples": count}
    spans = np.percentile(profile[valid], [10, 90])
    cumulative = np.r_[0.0, np.cumsum(profile * valid)]
    weights = np.r_[0, np.cumsum(valid)]
    windows = []
    for requested in config.broad_window_native_px:
        radius = max(1, math.ceil(requested / step))
        centers = np.arange(radius, len(profile) - radius + 1)
        if not len(centers):
            windows.append({"requested_side_width_native_px": requested, "status": "unassessed",
                            "reason": "window_exceeds_profile"})
            continue
        left_n = weights[centers] - weights[centers - radius]
        right_n = weights[centers + radius] - weights[centers]
        support = (left_n >= 0.8 * radius) & (right_n >= 0.8 * radius)
        if not support.any():
            windows.append({"requested_side_width_native_px": requested, "status": "unassessed",
                            "reason": "insufficient_two_side_support"})
            continue
        left = (cumulative[centers] - cumulative[centers - radius]) / np.maximum(left_n, 1)
        right = (cumulative[centers + radius] - cumulative[centers]) / np.maximum(right_n, 1)
        differences = right - left
        candidates = np.flatnonzero(support)
        peak = int(candidates[np.argmax(np.abs(differences[candidates]))])
        position = int(centers[peak] * step)
        width = radius * step
        height_native, width_native = shape
        box = ([max(0, position - width), 0, min(width_native, position + width), height_native]
               if axis == "x" else
               [0, max(0, position - width), width_native, min(height_native, position + width)])
        windows.append({"requested_side_width_native_px": requested, "actual_side_width_native_px": width,
                        "status": "measured", "eligible_centers": int(support.sum()),
                        "largest_two_side_difference_log1p": float(differences[peak]),
                        "position_native_px": position, "context_strip_level0_xyxy": box,
                        "interpretation": "descriptive maximum; not an artifact detection"})
    return {"status": "measured", "supported_samples": count,
            "support_fraction": float(valid.mean()), "p90_minus_p10_log1p": float(spans[1] - spans[0]),
            "windows": windows}


def _residual(levels, mask, period, step, reduction):
    height, width = levels.shape
    small_shape = (max(1, math.ceil(width / reduction)), max(1, math.ceil(height / reduction)))
    weight = cv2.resize(mask.astype(np.float32), small_shape, interpolation=cv2.INTER_AREA)
    value = cv2.resize(np.where(mask, levels, 0), small_shape, interpolation=cv2.INTER_AREA)
    sigma_x = max(0.5, 0.5 * period / step * small_shape[0] / width)
    sigma_y = max(0.5, 0.5 * period / step * small_shape[1] / height)
    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    value = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    trend = cv2.resize(value / np.maximum(weight, 1e-6), (width, height), interpolation=cv2.INTER_LINEAR)
    return levels - trend


def _fold(profile, valid, positions, period, step, config):
    if period / step < 2 * config.phase_bins:
        return {"status": "unassessed", "reason": "insufficient_samples_per_period", "period_native_px": period}
    phases = np.floor((positions % period) / period * config.phase_bins).astype(int)
    cycles = np.floor(positions / period).astype(int)
    selected = [c for c in np.unique(cycles[valid])
                if (valid & (cycles == c)).sum() >= config.min_cycle_samples
                and len(np.unique(phases[valid & (cycles == c)])) >= 0.5 * config.phase_bins]
    if len(selected) < config.min_cycles:
        return {"status": "unassessed", "reason": "insufficient_supported_cycles",
                "period_native_px": period, "supported_cycles": len(selected)}
    eligible = valid & np.isin(cycles, selected)
    sse = sst = 0.0
    tested = 0
    for cycle in selected:
        train, test = eligible & (cycles != cycle), eligible & (cycles == cycle)
        mean = float(profile[train].mean())
        sums = np.bincount(phases[train], weights=profile[train], minlength=config.phase_bins)
        counts = np.bincount(phases[train], minlength=config.phase_bins)
        prediction = np.where(counts > 0, sums / np.maximum(counts, 1), mean)
        sse += float(np.square(profile[test] - prediction[phases[test]]).sum())
        sst += float(np.square(profile[test] - mean).sum())
        tested += int(test.sum())
    sums = np.bincount(phases[eligible], weights=profile[eligible], minlength=config.phase_bins)
    counts = np.bincount(phases[eligible], minlength=config.phase_bins)
    fold = sums[counts > 0] / counts[counts > 0]
    r2 = 1 - sse / sst if sst > 1e-10 else None
    return {"status": "measured", "period_native_px": period, "supported_cycles": len(selected),
            "supported_phase_bins": int((counts > 0).sum()), "tested_profile_samples": tested,
            "within_image_cycle_prediction_r2": float(r2) if r2 is not None else None,
            "r2_status": "defined" if r2 is not None else "undefined_for_constant_profile",
            "fold_peak_to_peak_log1p": float(fold.max() - fold.min())}


def measure_brightness_patterns(native, *, config=None, region_mask=None):
    """Measure broad/periodic variation; optional native mask remains unreviewed.

    The default mask is a provisional foreground heuristic. Its identity is not
    established as the intended core. Supplied masks must be boolean native YX
    arrays; supplying one does not establish that its boundaries were reviewed.
    """
    config = BrightnessConfig() if config is None else config
    if not isinstance(config, BrightnessConfig):
        raise ValueError("A BrightnessConfig is required")
    if (not isinstance(native, np.ndarray) or isinstance(native, np.ma.MaskedArray)
            or native.ndim != 2 or not native.size
            or native.dtype.kind != "u" or native.dtype.itemsize != 2
            or native.size > MAX_NATIVE_PIXELS):
        raise ValueError("Expected a bounded, nonempty uint16 YX array")
    if region_mask is not None and (not isinstance(region_mask, np.ndarray)
                                   or isinstance(region_mask, np.ma.MaskedArray)
                                   or region_mask.dtype != np.bool_ or region_mask.shape != native.shape):
        raise ValueError("Region mask must be a native-shape boolean array")
    step = max(1, math.ceil(max(native.shape) / config.max_overview_dimension))
    overview = _block_mean(native, step)
    levels = np.log1p(overview)
    if region_mask is None:
        if config.foreground_method == "region_envelope":
            mask = np.ones_like(levels, bool)
            mask_method = "unreviewed_rectangle_envelope_including_dark_pixels"
        elif config.foreground_method == "hysteresis":
            mask, _, _ = axial.tissue_mask(overview)
            mask_method = "provisional_foreground_hysteresis"
        else:
            mask = _intensity_region(levels)
            mask_method = "provisional_nonzero_intensity_otsu"
    else:
        mask = _block_mean(region_mask, step) >= 0.99
        mask_method = "caller_supplied_unreviewed_region"
    if config.erosion_cells:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * config.erosion_cells + 1,) * 2, np.uint8)).astype(bool)
    residual = (_residual(levels, mask, config.period_native_px, step, config.trend_reduction)
                if config.period_native_px is not None and mask.any() else None)
    axes = {}
    for axis, length in (("x", native.shape[1]), ("y", native.shape[0])):
        profile, valid = _profile(levels, mask, axis, config.min_profile_coverage)
        broad = _broad_summary(profile, valid, axis, native.shape, step, config)
        if residual is None:
            periodic = {"status": "unassessed", "reason": "period_not_supplied" if config.period_native_px is None
                        else "insufficient_region_support"}
        else:
            values, supported = _profile(residual, mask, axis, config.min_profile_coverage)
            starts = np.arange(0, length, step)
            positions = starts + (np.minimum(step, length - starts) - 1) / 2
            requested = _fold(values, supported, positions, config.period_native_px, step, config)
            controls = [_fold(values, supported, positions, config.period_native_px * factor, step, config)
                        for factor in config.control_period_factors]
            r2_controls = [row["within_image_cycle_prediction_r2"] for row in controls
                           if row["status"] == "measured" and row["within_image_cycle_prediction_r2"] is not None]
            amplitude_controls = [row["fold_peak_to_peak_log1p"] for row in controls if row["status"] == "measured"]
            r2 = requested.get("within_image_cycle_prediction_r2")
            periodic = {"status": requested["status"], "requested": requested, "controls": controls,
                        "r2_minus_best_control": float(r2 - max(r2_controls)) if r2 is not None and r2_controls else None,
                        "median_control_amplitude_log1p": float(np.median(amplitude_controls)) if amplitude_controls else None,
                        "interpretation": "within-image repetition evidence; not independent artifact validation"}
        axes[axis] = {"broad": broad, "periodic": periodic}
    return {
        "schema": "fluoinspect.brightness-patterns.v1", "engine_id": ENGINE_ID,
        "configuration": config.to_dict(), "source_shape_yx": list(native.shape),
        "source_region_level0_xyxy": [0, 0, native.shape[1], native.shape[0]],
        "overview_step_native_px": step, "overview_shape_yx": list(overview.shape),
        "region_mask_method": mask_method, "region_mask_reviewed": False,
        "eligible_overview_fraction": float(mask.mean()), "axes": axes,
        "assessment": "experimental_measurements_only", "artifact_accuracy_validated": False,
        "quality_decision": None, "source_pixels_modified": False,
        "caveats": [
            "Foreground/region identity is unreviewed and may include neighbouring tissue or omit dim tissue.",
            "Natural tissue structure can produce broad or repeating brightness variation.",
            "Supplied period metadata or hypotheses are not independently verified acquisition geometry.",
            "Within-image prediction and alternative periods do not establish causal stitching/correction errors.",
            "Profile extrema propose context strips; no calibrated artifact or acceptance threshold is applied.",
            "Broad profiles are axis-aligned; oblique-pattern localization requires additional methods.",
        ],
    }
