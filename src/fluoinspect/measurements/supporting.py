"""Supporting measurements for a reconciled AF export screen.

The input is an already decoded uint16 grayscale plane. This module performs no
I/O, image correction, artifact classification or automatic quality acceptance.
Coordinates are native XY; array indexing is YX. Metrics are computed from source
values, while an Otsu mask on a bounded point-sampled overview supplies a separate,
unvalidated foreground estimate. The input array is never written or recast.

Background variation cannot establish background-removal damage. Relative detail
cannot exclude global blur, and dtype-max clipping is not a calibrated sensor test.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _percentile(histogram: np.ndarray, q: float) -> float:
    """Exact linear-interpolated percentile from a uint16 histogram."""
    n = int(histogram.sum())
    rank = (n - 1) * q / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    cumulative = np.cumsum(histogram)
    a = int(np.searchsorted(cumulative, lower + 1))
    b = int(np.searchsorted(cumulative, upper + 1))
    return float(a + (b - a) * (rank - lower))


def _raw_statistics(histogram: np.ndarray) -> dict:
    n = int(histogram.sum())
    values = np.arange(65536, dtype=np.float64)
    occupied = np.flatnonzero(histogram)
    mean = float(np.dot(histogram, values) / n)
    variance = float(np.dot(histogram, (values - mean) ** 2) / n)
    return {
        "pixel_count": n,
        "min": int(occupied[0]),
        "max": int(occupied[-1]),
        "mean": mean,
        "std_population": math.sqrt(max(variance, 0.0)),
        "p1": _percentile(histogram, 1),
        "median": _percentile(histogram, 50),
        "p99_5": _percentile(histogram, 99.5),
        "zero_count": int(histogram[0]),
        "zero_fraction": float(histogram[0] / n),
        "storage_clipped_count": int(histogram[-1]),
        "storage_clipped_fraction": float(histogram[-1] / n),
        "storage_clipping_value": 65535,
    }


def _foreground(native: np.ndarray, max_dim: int) -> tuple:
    step = max(1, math.ceil(max(native.shape) / max_dim))
    overview = native[::step, ::step]
    lo, hi = (float(v) for v in np.percentile(overview, [1, 99.5]))
    foreground = np.zeros(overview.shape, dtype=np.bool_)
    threshold = None
    if hi > lo:
        scaled = np.clip((overview.astype(np.float32) - lo) / (hi - lo), 0, 1)
        display = np.rint(scaled * 255).astype(np.uint8)
        threshold, mask = cv2.threshold(
            display, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        foreground = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        ).astype(np.bool_)
    return overview, foreground, {
        "method": "point-sampled Otsu then 5x5 closing; unvalidated",
        "sampling_step_native_pixels": step,
        "overview_shape_yx": list(overview.shape),
        "sampled_scale_p1": lo,
        "sampled_scale_p99_5": hi,
        "otsu_threshold_scaled_0_255": threshold,
        "scope": "provisional fluorescence foreground, not a reviewed core boundary",
    }


def _detail(patch: np.ndarray, foreground: np.ndarray) -> dict:
    result = {
        "foreground_pixel_count": int(foreground.sum()),
        "foreground_variance_raw": None,
        "derivative_support_count": 0,
        "laplacian_energy_raw": None,
        "normalized_detail": None,
    }
    if foreground.any():
        variance = float(np.var(patch[foreground], dtype=np.float64))
        result["foreground_variance_raw"] = variance
    else:
        return result
    if min(patch.shape) < 3:
        return result
    support = (
        foreground[1:-1, 1:-1]
        & foreground[:-2, 1:-1]
        & foreground[2:, 1:-1]
        & foreground[1:-1, :-2]
        & foreground[1:-1, 2:]
    )
    result["derivative_support_count"] = int(support.sum())
    if not support.any():
        return result
    values = patch.astype(np.float32)
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4 * values[1:-1, 1:-1]
    )
    selected = laplacian[support].astype(np.float64)
    energy = float(np.mean(selected * selected))
    result["laplacian_energy_raw"] = energy
    if variance > 0:
        result["normalized_detail"] = energy / variance
    return result


def _background(overview: np.ndarray, foreground: np.ndarray, cells: int) -> dict:
    records = []
    y_bounds = np.linspace(0, overview.shape[0], min(cells, overview.shape[0]) + 1)
    x_bounds = np.linspace(0, overview.shape[1], min(cells, overview.shape[1]) + 1)
    for y0, y1 in zip(y_bounds[:-1].astype(int), y_bounds[1:].astype(int)):
        for x0, x1 in zip(x_bounds[:-1].astype(int), x_bounds[1:].astype(int)):
            mask = ~foreground[y0:y1, x0:x1]
            n = int(mask.sum())
            fraction = n / mask.size
            # A sampling eligibility rule, not an artifact threshold.
            eligible = n >= 32 and fraction >= 0.8
            records.append(
                {
                    "overview_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "background_sample_count": n,
                    "background_fraction": fraction,
                    "eligible_for_span": eligible,
                    "median_raw": float(np.median(overview[y0:y1, x0:x1][mask]))
                    if eligible
                    else None,
                }
            )
    medians = [r["median_raw"] for r in records if r["eligible_for_span"]]
    result = {
        "scope": "sampled values outside provisional fluorescence foreground",
        "cell_eligibility": "at least 32 background samples and 80% background",
        "eligibility_settings_status": "development, unvalidated",
        "eligible_cell_count": len(medians),
        "span_available": len(medians) >= 4,
        "p10_cell_median_raw": None,
        "p90_cell_median_raw": None,
        "p90_minus_p10_raw": None,
        "p90_minus_p10_over_sampled_range": None,
        "cells": records,
    }
    if len(medians) >= 4:
        p10, p90 = (float(v) for v in np.percentile(medians, [10, 90]))
        lo, hi = (float(v) for v in np.percentile(overview, [1, 99.5]))
        result.update(
            p10_cell_median_raw=p10,
            p90_cell_median_raw=p90,
            p90_minus_p10_raw=p90 - p10,
            p90_minus_p10_over_sampled_range=(p90 - p10) / (hi - lo)
            if hi > lo
            else None,
        )
    return result


def compute_supporting_metrics(
    native: np.ndarray,
    *,
    patch_size: int = 1024,
    overview_max_dim: int = 1024,
    background_cells_per_axis: int = 8,
) -> dict:
    """Return JSON-serializable measurements; preserve all input pixels/dtype.

    Native patch bounds partition the entire image, including partial edge
    patches. No interpolation or masking modifies quantitative values. The
    sampled foreground is mapped back by its recorded integer sampling cells;
    foreground-dependent measures therefore inherit segmentation uncertainty.
    Detail uses a five-point Laplacian whose centre and four neighbours are all
    inside the same patch and provisional foreground. It is a relative texture
    measure, not a calibrated defocus score. No defect labels are returned.
    """
    if not isinstance(native, np.ndarray):
        raise TypeError("native must be a NumPy array")
    if native.ndim != 2 or 0 in native.shape:
        raise ValueError("expected a nonempty two-dimensional grayscale plane")
    if native.dtype.kind != "u" or native.dtype.itemsize != 2:
        raise ValueError("expected unsigned 16-bit native pixels")
    for name, value in (
        ("patch_size", patch_size),
        ("overview_max_dim", overview_max_dim),
        ("background_cells_per_axis", background_cells_per_axis),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    overview, foreground, foreground_meta = _foreground(native, overview_max_dim)
    step = foreground_meta["sampling_step_native_pixels"]
    height, width = native.shape
    histogram = np.zeros(65536, dtype=np.int64)
    patches = []
    foreground_count = 0
    for y0 in range(0, height, patch_size):
        y1 = min(y0 + patch_size, height)
        for x0 in range(0, width, patch_size):
            x1 = min(x0 + patch_size, width)
            patch = native[y0:y1, x0:x1]
            counts = np.bincount(patch.ravel(), minlength=65536)
            histogram += counts
            mask = foreground[np.ix_(np.arange(y0, y1) // step, np.arange(x0, x1) // step)]
            detail = _detail(patch, mask)
            foreground_count += detail["foreground_pixel_count"]
            patches.append(
                {
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "shape_yx": [y1 - y0, x1 - x0],
                    "raw": _raw_statistics(counts),
                    "provisional_foreground_fraction": detail["foreground_pixel_count"] / patch.size,
                    "detail": detail,
                }
            )
    raw = _raw_statistics(histogram)
    if raw["pixel_count"] != native.size:
        raise RuntimeError("native patch grid did not cover every pixel exactly once")
    detail_values = [p["detail"]["normalized_detail"] for p in patches]
    detail_values = [v for v in detail_values if v is not None]
    foreground_meta.update(
        native_mapped_pixel_count=foreground_count,
        native_mapped_fraction=foreground_count / native.size,
    )
    return {
        "schema_version": "af-supporting-metrics-v1",
        "status": "measurements only; unvalidated for quality decisions",
        "source_shape_yx": [height, width],
        "source_dtype": native.dtype.str,
        "raw": raw,
        "patch_size_native_pixels": patch_size,
        "patch_count": len(patches),
        "patch_pixel_count": sum(p["raw"]["pixel_count"] for p in patches),
        "patches": patches,
        "provisional_foreground": foreground_meta,
        "detail_summary": {
            "valid_patch_count": len(detail_values),
            "median_normalized_detail": float(np.median(detail_values)) if detail_values else None,
            "scope": "within provisional foreground; no defocus threshold or label",
        },
        "background": _background(overview, foreground, background_cells_per_axis),
        "caveats": [
            "Dtype-max clipping is not a calibrated sensor/ADC saturation measurement.",
            "Zero-valued pixels alone do not identify missing data or over-subtraction.",
            "Otsu foreground can omit dim tissue, include debris, and include neighbouring cores.",
            "Relative detail depends on tissue texture/noise and cannot exclude global blur.",
            "Background span is sampled, provisional and cannot establish processing damage.",
            "Soft banding, folds, debris and geometric registration errors are not classified.",
            "No-candidate or uniform measurements do not establish a quality pass.",
        ],
    }
