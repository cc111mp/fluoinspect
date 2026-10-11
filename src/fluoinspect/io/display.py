"""Shared display windows in stored uint16 units; independent of QC measurements."""
from __future__ import annotations

import numpy as np


def histogram(native):
    """Count every native pixel, using bounded stripes and no overview sampling."""
    if isinstance(native, np.ma.MaskedArray):
        raise ValueError("Masked images cannot establish native pixel availability")
    array = np.asarray(native)
    if array.ndim != 2 or not array.size or array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise ValueError("A nonempty uint16 YX image is required")
    counts = np.zeros(65536, dtype=np.uint64)
    for start in range(0, array.shape[0], 256):
        counts += np.bincount(array[start:start + 256].ravel(), minlength=65536).astype(np.uint64)
    return counts


def _counts(counts):
    if isinstance(counts, np.ma.MaskedArray):
        raise ValueError("Masked histogram counts are unsupported")
    counts = np.asarray(counts)
    if (counts.shape != (65536,) or counts.dtype.kind not in "ui"
            or (counts < 0).any() or not counts.sum(dtype=np.uint64)):
        raise ValueError("A nonempty 65536-bin integer histogram is required")
    return counts


def percentile(counts, q):
    """Exact linear-interpolated percentile of the values represented by counts."""
    counts = _counts(counts)
    if not isinstance(q, (int, float)) or not np.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("Percentile must be finite and between 0 and 100")
    rank = (int(counts.sum(dtype=np.uint64)) - 1) * q / 100
    cumulative = counts.cumsum(dtype=np.uint64)
    lo = np.searchsorted(cumulative, np.floor(rank), side="right")
    hi = np.searchsorted(cumulative, np.ceil(rank), side="right")
    return float(lo + (hi - lo) * (rank - np.floor(rank)))


def shared_window(histograms, q=99.9):
    """Use black=0 and the largest native percentile of the supplied regions.

    Supply the whole-image histogram and each known review-crop histogram when
    preparing a fixed panel. A session with future crops uses its source histogram
    and retains that window; clipping is reported separately for every later view.
    """
    histograms = list(histograms)
    if not histograms:
        raise ValueError("At least one native histogram is required")
    return [0, max(1, int(np.ceil(max(percentile(h, q) for h in histograms))))]


def validate_window(window):
    try:
        low, high = map(float, window)
    except (TypeError, ValueError):
        raise ValueError("Require two finite limits: 0 <= black < white <= 65535") from None
    if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high <= 65535):
        raise ValueError("Require two finite limits: 0 <= black < white <= 65535")
    return low, high


def render(native, window, mode="linear"):
    """Render uint16 values or native-unit block means without modifying inputs.

    The same native value maps to the same gray in every region using this window.
    Log is an explicit display option. Neither transfer function feeds a detector.
    """
    if isinstance(native, np.ma.MaskedArray):
        raise ValueError("Masked images cannot establish native pixel availability")
    low, high = validate_window(window)
    if mode not in {"linear", "log"}:
        raise ValueError("Display mode must be linear or log")
    array = np.asarray(native)
    if array.ndim != 2 or not array.size or array.dtype.kind not in "uif":
        raise ValueError("A nonempty numerical YX image in stored intensity units is required")
    result = np.empty(array.shape, dtype=np.uint8)
    for start in range(0, array.shape[0], 256):
        values = array[start:start + 256].astype(np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 65535).any():
            raise ValueError("Display values must be finite and between 0 and 65535")
        if mode == "linear":
            scaled = (values - low) / (high - low)
        else:
            scaled = (np.log1p(values) - np.log1p(low)) / (np.log1p(high) - np.log1p(low))
        result[start:start + 256] = np.floor(np.clip(scaled, 0, 1) * 255 + .5).astype(np.uint8)
    return result


def clipping(counts, window):
    """Report original pixels strictly outside the window, before 8-bit rounding."""
    counts = _counts(counts)
    low, high = validate_window(window)
    values = np.arange(65536)
    total = int(counts.sum(dtype=np.uint64))
    return {"below_percent": float(counts[values < low].sum(dtype=np.uint64) / total * 100),
            "above_percent": float(counts[values > high].sum(dtype=np.uint64) / total * 100)}
