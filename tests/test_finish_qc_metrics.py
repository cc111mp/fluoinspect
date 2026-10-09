"""Integrity and metric-behaviour tests, not AF diagnostic validation."""

import json

import numpy as np
import pytest


from fluoinspect.measurements.supporting import compute_supporting_metrics


def test_partial_patches_cover_exact_source_and_preserve_pixels():
    rng = np.random.default_rng(104)
    source = rng.integers(0, 65536, (1031, 2053), dtype=np.uint16)
    before = source.tobytes()
    source.setflags(write=False)
    metrics = compute_supporting_metrics(source)
    assert metrics["patch_count"] == 6
    assert metrics["patch_pixel_count"] == source.size
    cover = np.zeros(source.shape, dtype=np.uint8)
    for p in metrics["patches"]:
        x0, y0, x1, y1 = p["bbox_xyxy"]
        cover[y0:y1, x0:x1] += 1
        native = source[y0:y1, x0:x1]
        assert p["shape_yx"] == list(native.shape)
        assert p["raw"]["pixel_count"] == native.size
        assert p["raw"]["mean"] == pytest.approx(float(native.mean()))
    assert np.all(cover == 1)
    assert source.dtype == np.uint16
    assert source.tobytes() == before
    assert metrics["raw"]["mean"] == pytest.approx(float(source.mean()))
    assert metrics["raw"]["std_population"] == pytest.approx(float(source.std()))
    for key, percentile in (("p1", 1), ("median", 50), ("p99_5", 99.5)):
        assert metrics["raw"][key] == pytest.approx(float(np.percentile(source, percentile)))
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("value,zero_fraction,clipped_fraction", [(0, 1, 0), (65535, 0, 1), (5000, 0, 0)])
def test_uniform_planes_report_clipping_without_a_foreground(value, zero_fraction, clipped_fraction):
    source = np.full((130, 160), value, dtype=np.uint16)
    source.setflags(write=False)
    result = compute_supporting_metrics(source, patch_size=64)
    assert result["raw"]["mean"] == value
    assert result["raw"]["std_population"] == 0
    assert result["raw"]["zero_fraction"] == zero_fraction
    assert result["raw"]["storage_clipped_fraction"] == clipped_fraction
    assert result["provisional_foreground"]["native_mapped_fraction"] == 0
    assert result["detail_summary"]["valid_patch_count"] == 0
    assert result["background"]["p90_minus_p10_raw"] == 0
    assert result["background"]["p90_minus_p10_over_sampled_range"] is None
    assert all(p["detail"]["normalized_detail"] is None for p in result["patches"])
    json.dumps(result, allow_nan=False)


def test_big_endian_noncontiguous_input_is_preserved():
    original = np.arange(512, dtype=">u2").reshape(16, 32)
    source = original[:, ::2]
    source.setflags(write=False)
    before = original.tobytes()
    result = compute_supporting_metrics(source, patch_size=7)
    assert result["source_dtype"] == ">u2"
    assert result["raw"]["mean"] == pytest.approx(float(source.mean()))
    assert result["patch_pixel_count"] == source.size
    assert original.tobytes() == before
    assert source.dtype.str == ">u2"


def test_foreground_detail_distinguishes_texture_from_constant_tissue():
    source = np.zeros((256, 256), dtype=np.uint16)
    source[16:240, 16:240] = 10000
    flat = compute_supporting_metrics(source, patch_size=128)
    yy, xx = np.indices((224, 224))
    source[16:240, 16:240] = 10000 + ((yy + xx) % 2) * 2000
    textured = compute_supporting_metrics(source, patch_size=128)
    assert 0.6 < textured["provisional_foreground"]["native_mapped_fraction"] < 0.9
    assert textured["detail_summary"]["median_normalized_detail"] > 0
    # A uniform tissue patch has zero variance and therefore no valid ratio.
    assert flat["detail_summary"]["median_normalized_detail"] is None
    assert all(p["detail"]["derivative_support_count"] > 0 for p in textured["patches"])


def test_background_nonuniformity_is_a_raw_measurement_without_an_issue_label():
    source = np.tile(np.arange(128, dtype=np.uint16), (128, 1))
    result = compute_supporting_metrics(source, patch_size=64)
    background = result["background"]
    assert background["eligible_cell_count"] >= 4
    assert background["p90_minus_p10_raw"] > 0
    assert "issues" not in result
    assert "quality_decision" not in result
    assert "nonuniformity_candidate" not in background


def test_small_image_keeps_native_metrics_when_detail_background_are_unsupported():
    result = compute_supporting_metrics(np.array([[0, 65535]], dtype=np.uint16))
    assert result["patch_count"] == 1
    assert result["raw"]["zero_fraction"] == 0.5
    assert result["raw"]["storage_clipped_fraction"] == 0.5
    assert result["patches"][0]["detail"]["derivative_support_count"] == 0
    assert not result["background"]["span_available"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("array", [np.zeros((3, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint16), np.zeros((2, 2, 2), dtype=np.uint16)])
def test_unsupported_planes_are_rejected(array):
    with pytest.raises(ValueError):
        compute_supporting_metrics(array)


@pytest.mark.parametrize("option", [0, -1, True, 1.5])
def test_invalid_patch_size_is_rejected(option):
    with pytest.raises(ValueError):
        compute_supporting_metrics(np.zeros((4, 4), dtype=np.uint16), patch_size=option)
