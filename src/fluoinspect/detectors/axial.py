"""Native axis-aligned pattern screen for AF TIFF exports.

This revision fixes TIFF segment layout decoding and wide-window verification.
Candidate scores remain unvalidated screening evidence, not artifact confirmation
or a pass/fail decision. The source acquisition files must be staged read-only by
the caller; this module neither modifies them nor launches batch jobs.
"""

import hashlib

import cv2
import numpy as np
from ..io.native import (MAX_NATIVE_BYTES as MAX_NATIVE_BYTES,
                         MAX_NATIVE_PIXELS as MAX_NATIVE_PIXELS, read_native)

F = 8
EDGE = 12
CONFIRM_ROWS = 300
ENGINE_ID = "tile_qc_v2_20261008"
ENGINE_CHANGES = (
    "layout-aware TIFF decoder with semantic, segment and memory guards",
    "fine/wide verification at one center with complete wide-side support",
    "verified_pattern semantics and explicit central-location heuristic",
    "overlay at the verified native position and checked image write",
    "unique scoring representatives for overlapping native-position proposals",
)

SETTINGS = dict(
    block=F,
    edge_overview_px=EDGE,
    confirm_native_rows=CONFIRM_ROWS,
    tissue_level=0.35,
    tissue_low_level=0.12,
    cut_gap=12,
    cut_min_len=40,
    cut_min_cover=0.5,
    step_strip=3,
    step_window=96,
    step_pixel=0.15,
    step_agree=0.8,
    step_median=0.25,
    core_radius=0.7,
    core_border_overview_px=40,
    verify_half_width=40,
    verify_side=4,
    verify_min_jump=0.06,
    verify_tol=1,
    verify_wide_side=16,
    verify_wide_jump=0.04,
)




def load_overview(path):
    """Native pixels, block-mean overview and normalized pixel-buffer SHA-256."""
    a = read_native(path)
    view, pixel_sha = _overview_and_hash(a)
    return a, view, pixel_sha


def _overview_and_hash(a):
    """Normalize integer byte order before hashing; retain all remainder pixels."""
    if a.ndim != 2 or a.dtype.kind != "u" or a.dtype.itemsize != 2:
        raise ValueError("expected a single-plane uint16 native array")
    if a.size > MAX_NATIVE_PIXELS or a.size * 2 > MAX_NATIVE_BYTES:
        raise ValueError("native decoded image exceeds configured memory limit")
    a = np.ascontiguousarray(a, dtype=np.uint16)
    height, width = a.shape
    h, w = height // F, width // F
    if h == 0 or w == 0:
        raise ValueError("image is smaller than an overview block")
    view = np.empty((h, w), np.float32)
    digest = hashlib.sha256()
    for y in range(0, height, 64 * F):
        block = np.asarray(a[y : min(height, y + 64 * F)])
        digest.update(block.tobytes())
        rows = min(h, (y + block.shape[0]) // F) - y // F
        if rows > 0:
            b = block[: rows * F, : w * F].astype(np.float32)
            view[y // F : y // F + rows] = b.reshape(rows, F, w, F).mean((1, 3))
    return view, digest.hexdigest()


def tissue_mask(view):
    """Provisional hysteresis mask; tissue identity/boundaries are unvalidated."""
    levels = cv2.GaussianBlur(np.log1p(view), (0, 0), 1.5)
    background, high = np.percentile(levels, 30), np.percentile(levels, 97)
    strong = levels > background + SETTINGS["tissue_level"] * (high - background)
    low = background + SETTINGS["tissue_low_level"] * (high - background)
    n, labels = cv2.connectedComponents((levels > low).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    contrast = float(np.expm1(high) - np.expm1(background))
    return keep[labels], float(np.expm1(low)), max(contrast, 1.0)


def _dedupe(segments):
    segments.sort(key=lambda r: -r["length"])
    keep = []
    for row in segments:
        if all(
            abs(row["pos"] - retained["pos"]) > 3
            or row["end"] <= retained["start"]
            or row["start"] >= retained["end"]
            for retained in keep
        ):
            keep.append(row)
    return keep


def cut_candidates(tissue):
    """Axis-aligned mask-boundary runs; boundaries need native verification."""
    gap = SETTINGS["cut_gap"]
    minimum_length = SETTINGS["cut_min_len"]
    minimum_coverage = SETTINGS["cut_min_cover"]
    _, width = tissue.shape
    transitions = np.diff(tissue.astype(np.int8), axis=1)
    segments = []
    for direction in (1, -1):
        mask = transitions == direction
        band = mask.copy()
        band[:, 1:] |= mask[:, :-1]
        band[:, :-1] |= mask[:, 1:]
        for x in range(EDGE, width - EDGE):
            indices = np.flatnonzero(band[:, x])
            if len(indices) < minimum_length * minimum_coverage:
                continue
            start = previous = indices[0]
            for index in list(indices[1:]) + [None]:
                if index is None or index - previous > gap + 1:
                    if (
                        previous - start + 1 >= minimum_length
                        and band[start : previous + 1, x].mean() >= minimum_coverage
                    ):
                        segments.append(
                            dict(
                                pos=x + 1,
                                start=int(start),
                                end=int(previous + 1),
                                length=int(previous - start + 1),
                                source="outline",
                            )
                        )
                    if index is not None:
                        start = index
                if index is not None:
                    previous = index
    return segments


def step_candidates(levels, tissue):
    """Overview same-direction brightness steps with provisional tissue support."""
    side = SETTINGS["step_strip"]
    window = SETTINGS["step_window"]
    height, width = levels.shape
    cumulative = np.cumsum(np.pad(levels, ((0, 0), (1, 0))), axis=1)
    cumulative_tissue = np.cumsum(
        np.pad(tissue.astype(np.float32), ((0, 0), (1, 0))), axis=1
    )
    xs = np.arange(side, width - side + 1)
    step = (
        cumulative[:, xs + side] - 2 * cumulative[:, xs] + cumulative[:, xs - side]
    ) / side
    left = (cumulative_tissue[:, xs] - cumulative_tissue[:, xs - side]) / side
    right = (cumulative_tissue[:, xs + side] - cumulative_tissue[:, xs]) / side
    valid = ((left > 0.5) | (right > 0.5)).astype(np.float32)
    kernel = np.ones((window, 1), np.float32)

    def filtered(a):
        return cv2.filter2D(
            a.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
        )

    denominator = filtered(valid)
    positive = filtered((step > SETTINGS["step_pixel"]) * valid) / np.maximum(
        denominator, 1
    )
    negative = filtered((step < -SETTINGS["step_pixel"]) * valid) / np.maximum(
        denominator, 1
    )
    mean = filtered(step * valid) / np.maximum(denominator, 1)
    supported = denominator >= 0.8 * window
    segments = []
    for hit in (
        supported
        & (positive >= SETTINGS["step_agree"])
        & (mean > SETTINGS["step_median"]),
        supported
        & (negative >= SETTINGS["step_agree"])
        & (mean < -SETTINGS["step_median"]),
    ):
        for j in np.flatnonzero(hit.any(0)):
            x = int(xs[j])
            if x < EDGE or x > width - EDGE:
                continue
            rows = np.flatnonzero(hit[:, j])
            start = previous = rows[0]
            for row in list(rows[1:]) + [None]:
                if row is None or row - previous > 8:
                    a = max(0, start - window // 2)
                    b = min(height, previous + window // 2)
                    segments.append(
                        dict(
                            pos=x,
                            start=int(a),
                            end=int(b),
                            length=int(b - a),
                            source="step",
                        )
                    )
                    if row is not None:
                        start = row
                if row is not None:
                    previous = row
    return segments


def verify(native, candidate, vertical, low_raw, contrast):
    """Count native rows supporting the same jump position at both side widths.

    Supporting rows can be disjoint and jump direction may vary. A verified
    straight intensity pattern does not establish its biological/acquisition cause.
    """
    half = SETTINGS["verify_half_width"]
    side = SETTINGS["verify_side"]
    wide_side = SETTINGS["verify_wide_side"]
    c0 = candidate["pos"] * F
    a, b = candidate["start"] * F, candidate["end"] * F
    if vertical:
        lo, hi = max(0, c0 - half), min(native.shape[1], c0 + half)
        strip = np.asarray(native[a:b, lo:hi], np.float32)
    else:
        lo, hi = max(0, c0 - half), min(native.shape[0], c0 + half)
        strip = np.asarray(native[lo:hi, a:b], np.float32).T
    unverified = dict(exact_rows=0, native_line=None, median_jump=0.0)
    support = max(side, wide_side)
    if strip.shape[1] < 2 * support or strip.shape[0] < 20:
        return unverified
    if not np.isfinite(contrast) or contrast <= 0:
        raise ValueError("native verification contrast must be finite and positive")
    strip = cv2.blur(strip, (1, 3))
    cumulative = np.cumsum(np.pad(strip, ((0, 0), (1, 0))), axis=1)
    # Every eligible center has enough pixels for both the fine and wide means.
    # Never clamp a detected jump to another position for wide verification.
    j = np.arange(support, strip.shape[1] - support + 1)
    left = (cumulative[:, j] - cumulative[:, j - side]) / side
    right = (cumulative[:, j + side] - cumulative[:, j]) / side
    difference = np.abs(right - left) / contrast
    maximum, argmaximum = difference.max(1), difference.argmax(1)
    rows = np.arange(len(difference))
    bright = np.maximum(left, right)[rows, argmaximum]
    x = j[argmaximum]
    wide_left = (cumulative[rows, x] - cumulative[rows, x - wide_side]) / wide_side
    wide_right = (cumulative[rows, x + wide_side] - cumulative[rows, x]) / wide_side
    wide = np.abs(wide_right - wide_left) / contrast
    useful = (
        (maximum > SETTINGS["verify_min_jump"])
        & (bright > low_raw)
        & (wide > SETTINGS["verify_wide_jump"])
    )
    if useful.sum() < 20:
        return unverified
    mode = np.bincount(argmaximum[useful]).argmax()
    exact = useful & (np.abs(argmaximum - mode) <= SETTINGS["verify_tol"])
    return dict(
        exact_rows=int(exact.sum()),
        native_line=int(lo + j[mode]),
        median_jump=float(np.median(maximum[exact])) if exact.any() else 0.0,
    )


def screen(path):
    """Screen an export; verified flags refer only to pattern verification."""
    return screen_native(read_native(path))


def screen_native(native):
    """Screen an already-decoded array without another read or modifying pixels.

    Return the same tuple as screen(): overview, provisional mask, line candidates,
    normalized pixel-buffer hash and native YX shape. All overview candidates and
    their raw verification evidence remain in lines. For counts or scoring, select
    unique_verified_pattern, rather than verified_pattern: several overview
    proposals can converge to the same native boundary.
    """
    view, pixel_sha = _overview_and_hash(native)
    tissue, low_raw, contrast = tissue_mask(view)
    levels = np.log1p(view)
    lines = []
    for vertical, mask, transformed in (
        (True, tissue, levels),
        (False, tissue.T.copy(), levels.T.copy()),
    ):
        candidates = _dedupe(cut_candidates(mask) + step_candidates(transformed, mask))
        for candidate in candidates:
            candidate.update(verify(native, candidate, vertical, low_raw, contrast))
            candidate["orientation"] = "vertical" if vertical else "horizontal"
            candidate["verified_pattern"] = candidate["exact_rows"] >= CONFIRM_ROWS
            # Compatibility with old consumers; never means confirmed artifact.
            candidate["confirmed"] = candidate["verified_pattern"]
            lines.append(candidate)
    _mark_unique_verified_patterns(lines)
    return view, tissue, lines, pixel_sha, native.shape


def _mark_unique_verified_patterns(lines):
    """Keep the strongest scoring representative of overlapping native proposals.

    Native positions within the existing one-pixel verification tolerance are
    treated as duplicates only for the same orientation and overlapping candidate
    spans. Separate spans and crossing orientations remain separate evidence.
    exact_rows and verified_pattern are unchanged. The representative's count is
    retained; an affected-area union cannot be inferred from aggregate row counts.
    """
    for line in lines:
        line["unique_verified_pattern"] = False
        line["duplicate_of_candidate_index"] = None

    def strength(index):
        line = lines[index]
        return (
            -line["exact_rows"],
            -line["median_jump"],
            -(line["end"] - line["start"]),
            line["orientation"],
            line["native_line"],
            line["start"],
            line["end"],
            line["source"],
            line["pos"],
            index,
        )

    verified = sorted(
        (i for i, line in enumerate(lines) if line["verified_pattern"]),
        key=strength,
    )
    representatives = []
    for index in verified:
        line = lines[index]
        duplicate = None
        for retained_index in representatives:
            retained = lines[retained_index]
            if (
                line["orientation"] == retained["orientation"]
                and abs(line["native_line"] - retained["native_line"])
                <= SETTINGS["verify_tol"]
                and max(line["start"], retained["start"])
                < min(line["end"], retained["end"])
            ):
                duplicate = retained_index
                break
        if duplicate is None:
            line["unique_verified_pattern"] = True
            representatives.append(index)
        else:
            # Zero-based index into this returned lines array, not a physical ID.
            line["duplicate_of_candidate_index"] = duplicate


def in_central_ellipse(line, shape):
    """Test a candidate midpoint against the original location heuristic.

    This does not prove that the line belongs to the target biological core, nor
    that excluded lines belong to neighbouring cores. Keep the original midpoint
    and border calculations so the scoring definition remains reproducible.
    """
    height, width = shape[0] // F, shape[1] // F
    border = SETTINGS["core_border_overview_px"]
    if line["orientation"] == "vertical":
        x = line["pos"]
        y = (line["start"] + line["end"]) / 2
        position, limit = line["pos"], width
    else:
        x = (line["start"] + line["end"]) / 2
        y = line["pos"]
        position, limit = line["pos"], height
    if position < border or position > limit - border:
        return False
    return ((x - width / 2) / (width / 2)) ** 2 + (
        (y - height / 2) / (height / 2)
    ) ** 2 <= SETTINGS["core_radius"] ** 2


def in_core(line, shape):
    """Compatibility alias for central location, not verified core membership."""
    return in_central_ellipse(line, shape)


def draw(view, lines, path):
    """Write a preview of verified patterns; red marks the location heuristic.

    Orange means a verified line was excluded by the location rule. Neither colour
    confirms an artifact cause or biological core identity. native_line defines
    the drawn axis; pos remains the overview proposal coordinate in the record.
    """
    high = np.percentile(view, 99.5) or 1
    display = (np.clip(view / high, 0, 1) ** 0.5 * 255).astype(np.uint8)
    canvas = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    for line in lines:
        if not line.get(
            "unique_verified_pattern",
            line.get("verified_pattern", line.get("confirmed", False)),
        ):
            continue
        native_position = line.get("native_line")
        position = (
            int(round(native_position / F))
            if native_position is not None
            else line["pos"]
        )
        if line["orientation"] == "vertical":
            p1, p2 = (position, line["start"]), (position, line["end"])
        else:
            p1, p2 = (line["start"], position), (line["end"], position)
        central = line.get("in_central_ellipse", line.get("in_core", False))
        cv2.line(canvas, p1, p2, (0, 0, 255) if central else (0, 165, 255), 4)
    scale = 1024 / max(canvas.shape[:2])
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write pattern overlay: {path}")
