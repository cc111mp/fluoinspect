"""Known-geometry and evidence-contract challenges for dark-band measurements."""

import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.detectors.bands import BandConfig, measure_alternating_bands
from fluoinspect.investigation.session import native_hash, sha
from fluoinspect.pipeline import measure_image
from fluoinspect.tools.evidence import _bind_record


def pair_image(angle=0, side=1600, half_width=8):
    y, x = np.indices((side, side))
    normal = -np.sin(np.deg2rad(angle)) * (x - side / 2) + np.cos(np.deg2rad(angle)) * (
        y - side / 2
    )
    return np.where(
        (np.abs(normal - 250) <= half_width) | (np.abs(normal + 250) <= half_width),
        0,
        4000,
    ).astype(np.uint16)


def narrow_config(**kwargs):
    options = {
        "widths_native_px": (4, 16, 64),
        "angles_degrees": (0, 90),
        "max_overview_dimension": 1024,
    }
    options.update(kwargs)
    return BandConfig(**options)


@pytest.mark.parametrize("angle", [0, 0.6, 15, 45, 90])
def test_two_dark_bands_and_intervening_bright_region(angle):
    native = pair_image(angle)
    before = native_hash(native)
    angles = (0, 90) if angle in (0, 0.6, 90) else (angle,)
    result = measure_alternating_bands(
        native, config=narrow_config(angles_degrees=angles)
    )
    assert len(result["native_verified_bands"]) == 2
    assert len(result["parallel_alternating_pairs"]) == 1
    assert len(result["straight_edged_alternating_pairs"]) == 1
    assert result["parallel_alternating_pairs"][0][
        "spacing_native_px"
    ] == pytest.approx(500, abs=10)
    for band in result["native_verified_bands"]:
        distance = abs((band["angle_degrees"] - angle + 90) % 180 - 90)
        assert distance < 0.3
        assert band["native_support_fraction"] > 0.95
        assert band["median_relative_intensity_drop"] > 0.8
        assert band["native_sampling_method"] == "nearest_original_pixels"
        assert band["median_exact_zero_fraction_in_band_samples"] > 0.8
        assert band["dark_edge_geometry"]["straight_edged_loss_pattern"] is True
    assert native_hash(native) == before
    assert result["dark_pixels_retained"] is True
    assert result["quality_decision"] is None
    assert result["artifact_accuracy_validated"] is False


def test_single_dark_band_is_not_reported_as_alternation():
    native = np.full((1200, 1200), 4000, np.uint16)
    native[592:608] = 0
    result = measure_alternating_bands(native, config=narrow_config())
    assert len(result["native_verified_bands"]) == 1
    assert not result["parallel_alternating_pairs"]


@pytest.mark.parametrize("kind", ["constant", "gradient", "hole", "curved_channel"])
def test_local_holes_and_monotone_changes_do_not_become_parallel_bands(kind):
    side = 1200
    y, x = np.indices((side, side))
    native = np.full((side, side), 4000, np.uint16)
    if kind == "gradient":
        native = np.broadcast_to(
            (2000 + np.arange(side) * 3).astype(np.uint16)[:, None], (side, side)
        ).copy()
    elif kind == "hole":
        native[(x - 600) ** 2 + (y - 600) ** 2 < 120**2] = 0
    elif kind == "curved_channel":
        native[np.abs(y - (600 + 180 * np.sin(x / 110))) < 6] = 0
    result = measure_alternating_bands(native, config=narrow_config())
    assert not result["parallel_alternating_pairs"]
    if kind in {"constant", "gradient", "hole"}:
        assert not result["native_verified_bands"]


def test_region_changes_scope_and_global_coordinates():
    native = np.full((1800, 1800), 4000, np.uint16)
    native[390:410] = 0
    native[890:910] = 0
    native[1390:1410] = 0
    result = measure_alternating_bands(
        native, config=narrow_config(region_xyxy=(100, 100, 1700, 1100))
    )
    assert result["source_region_level0_xyxy"] == [100, 100, 1700, 1100]
    assert len(result["native_verified_bands"]) == 2
    for band in result["native_verified_bands"]:
        box = band["context_bbox_level0_xyxy"]
        assert 100 <= box[0] < box[2] <= 1700 and 100 <= box[1] < box[3] <= 1100
    assert result["region_identity_reviewed"] is False


def test_unresolved_widths_and_blank_signal_are_explicit():
    result = measure_alternating_bands(
        np.full((1600, 1600), 2000, np.uint16),
        config=narrow_config(widths_native_px=(2,), max_overview_dimension=128),
    )
    assert result["screen_status"] == "unassessed"
    assert result["unassessed_widths"][0]["reason"] == "less_than_two_overview_samples"
    result = measure_alternating_bands(
        np.zeros((1200, 1200), np.uint16), config=narrow_config()
    )
    assert result["screen_status"] == "unassessed"
    assert all(w["reason"] == "no_positive_signal" for w in result["unassessed_widths"])


def test_black_image_edge_is_not_a_two_flank_band():
    native = np.full((1200, 1200), 4000, np.uint16)
    native[:100] = 0
    result = measure_alternating_bands(native, config=narrow_config())
    assert not result["native_verified_bands"]


def test_per_scale_candidate_omission_is_recorded():
    native = np.full((2400, 1200), 4000, np.uint16)
    for row in [400, 800, 1200, 1600, 2000]:
        native[row - 8 : row + 8] = 0
    result = measure_alternating_bands(
        native,
        config=narrow_config(
            widths_native_px=(16,), angles_degrees=(0,), max_overview_dimension=2048
        ),
    )
    assert result["per_orientation_width_proposals_truncated"] is True
    assert result["proposal_budget_truncated"] is True


@pytest.mark.parametrize(
    "options",
    [
        {"widths_native_px": [16]},
        {"widths_native_px": (0,)},
        {"widths_native_px": (True,)},
        {"angles_degrees": (0, 180)},
        {"angles_degrees": (90, -90)},
        {"angles_degrees": (math.nan,)},
        {"min_depth_log1p": math.inf},
        {"min_support_fraction": 0},
        {"max_proposals": True},
        {"region_xyxy": (10, 0, 5, 20)},
        {"region_xyxy": [0, 0, 20, 20]},
    ],
)
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        BandConfig(**options)


@pytest.mark.parametrize(
    "native",
    [
        np.zeros((10, 10), np.float32),
        np.zeros((10, 10), np.int16),
        np.zeros((1, 10, 10), np.uint16),
        np.zeros((0, 10), np.uint16),
    ],
)
def test_unsupported_input_rejected(native):
    with pytest.raises(ValueError):
        measure_alternating_bands(native)


def test_out_of_source_region_rejected():
    with pytest.raises(ValueError):
        measure_alternating_bands(
            np.ones((128, 128), np.uint16),
            config=BandConfig(region_xyxy=(0, 0, 129, 100)),
        )


def test_pipeline_opt_in_identity_and_packet_binding(tmp_path):
    source = tmp_path / "source.tif"
    native = pair_image(side=1200)
    tifffile.imwrite(source, native)
    baseline = measure_image(source)
    enabled = measure_image(source, band_config=narrow_config())
    assert baseline["alternating_bands"] is None
    assert "alternating_bands" not in baseline["measurement_configuration"]
    assert baseline["run_identity"] != enabled["run_identity"]
    path = tmp_path / "record.json"
    path.write_text(json.dumps(enabled))
    info = {"shape_yx": list(native.shape), "source_file_sha256": sha(source)}
    bound = _bind_record(path, sha(path), info, native_hash(native))
    assert len(bound["alternating_bands"]["parallel_alternating_pairs"]) == 1
    assert bound["artifact_detection_accuracy_validated"] is False
    for field, bad_value in [
        ("source_region_level0_xyxy", [0, 0, 500, 500]),
        ("quality_decision", "accept"),
        ("dark_pixels_retained", False),
    ]:
        bad = copy.deepcopy(enabled)
        bad["alternating_bands"][field] = bad_value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            _bind_record(path, sha(path), info, native_hash(native))
    missing = copy.deepcopy(enabled)
    missing["alternating_bands"] = None
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError):
        _bind_record(path, sha(path), info, native_hash(native))


def test_cli_retains_requested_band_region(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "source.tif"
    tifffile.imwrite(source, pair_image(side=1200))
    output = tmp_path / "measurements"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fluoinspect",
            "measure",
            str(source),
            "--output",
            str(output),
            "--alternating-bands",
            "--band-region",
            "10,10,1190,1190",
            "--band-widths-px",
            "4,16,64",
            "--band-angles-deg",
            "0,90",
        ],
        check=True,
        capture_output=True,
    )
    result = json.loads((output / "measurements.json").read_text())
    assert result["alternating_bands"]["source_region_level0_xyxy"] == [
        10,
        10,
        1190,
        1190,
    ]
    assert result["alternating_bands"]["quality_decision"] is None
