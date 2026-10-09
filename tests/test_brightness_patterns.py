"""Known-pattern numerical challenges and interpretation/integrity boundaries."""
import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.investigation import session
from fluoinspect.measurements.brightness import BrightnessConfig, _block_mean, measure_brightness_patterns
from fluoinspect.pipeline import measure_image
from fluoinspect.tools.evidence import build_packet


def stripe_image(width=1024, height=128, period=128):
    x = np.arange(width)
    signal = 1600 * np.exp(0.3 * np.sin(2 * np.pi * x / period))
    noise = np.random.default_rng(19).normal(0, 8, (height, width))
    return np.clip(signal[None, :] + noise, 0, 65535).astype(np.uint16)


def config(**kwargs):
    return BrightnessConfig(phase_bins=16, erosion_cells=0, **kwargs)


def measure(values, **kwargs):
    return measure_brightness_patterns(values, region_mask=np.ones_like(values, bool), **kwargs)


def test_repeated_brightness_predicts_other_cycles_and_beats_wrong_periods():
    values = stripe_image()
    before = values.copy()
    result = measure(values, config=config(period_native_px=128))
    periodic = result["axes"]["x"]["periodic"]
    requested = periodic["requested"]
    assert requested["status"] == "measured"
    assert requested["supported_cycles"] == 8
    assert requested["within_image_cycle_prediction_r2"] > 0.8
    assert requested["fold_peak_to_peak_log1p"] > 0.4
    assert periodic["r2_minus_best_control"] > 0.3
    assert np.array_equal(before, values)
    assert result["quality_decision"] is None and not result["artifact_accuracy_validated"]
    json.dumps(result, allow_nan=False)


def test_transposing_data_transposes_measurement_axes():
    values = stripe_image()
    a = measure(values, config=config(period_native_px=128))
    b = measure(values.T, config=config(period_native_px=128))
    for first, second in [("x", "y"), ("y", "x")]:
        original = a["axes"][first]["periodic"]["requested"]
        transposed = b["axes"][second]["periodic"]["requested"]
        for key in original:
            if key in {"within_image_cycle_prediction_r2", "fold_peak_to_peak_log1p"} and original[key] is not None:
                assert original[key] == pytest.approx(transposed[key], abs=1e-6)
            else:
                assert original[key] == transposed[key]


def test_period_must_be_explicit_and_broad_variation_remains_measured():
    values = np.tile(np.linspace(1000, 2000, 1024).astype(np.uint16), (128, 1))
    result = measure(values, config=config())
    broad = result["axes"]["x"]["broad"]
    assert broad["p90_minus_p10_log1p"] > 0.4
    assert max(abs(w["largest_two_side_difference_log1p"]) for w in broad["windows"]) > 0.1
    assert result["axes"]["x"]["periodic"]["reason"] == "period_not_supplied"
    for window in broad["windows"]:
        x0, y0, x1, y1 = window["context_strip_level0_xyxy"]
        assert 0 <= x0 < x1 <= 1024 and y0 == 0 and y1 == 128
    assert result["quality_decision"] is None


def test_broad_context_strip_spans_a_known_transition():
    values = np.full((128, 1024), 1000, np.uint16)
    values[:, 512:] = 2000
    result = measure(values, config=config())
    windows = result["axes"]["x"]["broad"]["windows"]
    for window in windows:
        assert window["position_native_px"] == 512
        assert window["context_strip_level0_xyxy"][0] < 512 < window["context_strip_level0_xyxy"][2]
    # A real intensity boundary is not proof of a stitching artifact.
    assert result["assessment"] == "experimental_measurements_only"


def test_nonrepeating_cycle_shapes_do_not_predict_each_other():
    rng = np.random.default_rng(28)
    profile = rng.normal(0, 0.15, 1024)
    values = np.tile(np.clip(1600 * np.exp(profile), 0, 65535).astype(np.uint16), (128, 1))
    result = measure(values, config=config(period_native_px=128))
    assert result["axes"]["x"]["periodic"]["requested"]["within_image_cycle_prediction_r2"] < 0.1


def test_constant_regions_have_finite_zero_amplitude_and_no_quality_acceptance():
    values = np.full((128, 1024), 1500, np.uint16)
    result = measure(values, config=config(period_native_px=128))
    requested = result["axes"]["x"]["periodic"]["requested"]
    assert requested["fold_peak_to_peak_log1p"] < 1e-5
    assert requested["within_image_cycle_prediction_r2"] is None
    assert result["quality_decision"] is None
    json.dumps(result, allow_nan=False)


def test_too_few_cycles_remain_unassessed():
    values = stripe_image(width=256)
    result = measure(values, config=config(period_native_px=128))
    requested = result["axes"]["x"]["periodic"]["requested"]
    assert requested["status"] == "unassessed" and requested["reason"] == "insufficient_supported_cycles"


def test_coarse_sampling_requires_a_finer_view_instead_of_a_negative_result():
    values = stripe_image(width=1024, period=64)
    result = measure(values, config=config(period_native_px=64, max_overview_dimension=64))
    requested = result["axes"]["x"]["periodic"]["requested"]
    assert requested["status"] == "unassessed" and requested["reason"] == "insufficient_samples_per_period"


def test_empty_and_sparse_regions_do_not_become_negative_qc_labels():
    values = stripe_image()
    for mask in (np.zeros_like(values, bool), np.arange(values.shape[0])[:, None] == np.zeros_like(values)):
        result = measure_brightness_patterns(values, config=config(period_native_px=128), region_mask=mask)
        assert result["axes"]["x"]["broad"]["status"] == "unassessed"
        assert result["quality_decision"] is None


def test_block_means_keep_constant_partial_edge_blocks_constant():
    values = np.full((65, 131), 500, np.uint16)
    result = _block_mean(values, 3)
    assert result.shape == (22, 44) and np.all(result == 500)


@pytest.mark.parametrize("options", [
    {"period_native_px": float("nan")}, {"period_native_px": float("inf")},
    {"period_native_px": -1}, {"period_native_px": True},
    {"min_cycles": 2}, {"min_profile_coverage": 0}, {"min_profile_coverage": float("nan")},
    {"control_period_factors": (1,)}, {"broad_window_native_px": (0,)},
    {"max_overview_dimension": 4096},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        BrightnessConfig(**options)


@pytest.mark.parametrize("values", [
    np.zeros((128, 128), np.float32), np.zeros((128, 128), np.int16),
    np.zeros((128, 128, 1), np.uint16), np.zeros((0, 128), np.uint16),
    np.ma.array(np.ones((128, 128), np.uint16)),
])
def test_unsupported_arrays_are_rejected(values):
    with pytest.raises(ValueError):
        measure_brightness_patterns(values)


def test_mask_shape_and_type_are_checked():
    values = stripe_image()
    for mask in [np.zeros((8, 8), bool), np.zeros_like(values), np.ma.array(np.ones_like(values, bool))]:
        with pytest.raises(ValueError):
            measure_brightness_patterns(values, region_mask=mask)


def test_brightness_mask_separates_known_intensity_populations_and_preserves_zeros():
    values = np.full((128, 1024), 150, np.uint16)
    values[:, 512:] = 3000
    values[:16] = 0
    result = measure_brightness_patterns(values, config=config(period_native_px=128))
    assert result["region_mask_method"] == "provisional_nonzero_intensity_otsu"
    assert result["eligible_overview_fraction"] == pytest.approx(0.4375)
    assert not result["region_mask_reviewed"] and result["quality_decision"] is None


def test_zero_or_constant_fields_leave_foreground_unassessed():
    for value in [0, 1500]:
        result = measure_brightness_patterns(np.full((128, 1024), value, np.uint16), config=config(period_native_px=128))
        assert result["eligible_overview_fraction"] == 0
        assert result["axes"]["x"]["periodic"]["reason"] == "insufficient_region_support"


def test_foreground_method_is_explicit_and_changes_configuration_identity(tmp_path):
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, stripe_image(), photometric="minisblack")
    first = measure_image(source, brightness_config=config(foreground_method="hysteresis"))
    second = measure_image(source, brightness_config=config(foreground_method="intensity_otsu"))
    assert first["run_identity"] != second["run_identity"]
    assert first["pixel_sha256"] == second["pixel_sha256"]


def test_cli_records_period_and_mask_assumptions_without_quality_labels(tmp_path):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    source = source_dir / "source.tif"
    tifffile.imwrite(source, stripe_image(), photometric="minisblack")
    output = tmp_path / "output"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "measure", str(source),
                              "--output", str(output), "--pattern-period-px", "128",
                              "--pattern-period-basis", "hypothesis",
                              "--pattern-foreground-method", "hysteresis"],
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    record = json.loads((output / "measurements.json").read_text())
    settings = record["brightness_patterns"]["configuration"]
    assert settings["period_native_px"] == 128 and settings["period_basis"] == "hypothesis"
    assert settings["foreground_method"] == "hysteresis"
    assert record["human_decision"] == "" and record["brightness_patterns"]["quality_decision"] is None


def test_period_configuration_changes_identity_without_changing_pixels(tmp_path):
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, stripe_image(), photometric="minisblack")
    a = measure_image(source, brightness_config=config(period_native_px=128))
    b = measure_image(source, brightness_config=config(period_native_px=192))
    assert a["pixel_sha256"] == b["pixel_sha256"]
    assert a["run_identity"] != b["run_identity"]
    assert a["brightness_patterns"]["configuration"]["period_native_px"] == 128
    assert a["human_decision"] == "" and a["automated_assessment"] == "measurements_only"


def test_new_measurements_reach_bound_evidence_and_legacy_absence_is_explicit(tmp_path):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    source = source_dir / "source.tif"
    tifffile.imwrite(source, stripe_image(), photometric="minisblack")
    inspection = tmp_path / "inspection"
    session.create_session(source, inspection)
    record = measure_image(source, brightness_config=config(period_native_px=128))
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(record))
    packet = build_packet(inspection, record=path, expected_record_sha256=session.sha(path))
    assert packet["measurements"]["brightness_patterns"]["schema"] == "fluoinspect.brightness-patterns.v1"
    assert packet["human_QC_decision"] == "" and not packet["verification"]["artifact_detection_accuracy_validated"]
    del record["brightness_patterns"]
    path.write_text(json.dumps(record))
    legacy = build_packet(inspection, record=path, expected_record_sha256=session.sha(path))
    assert legacy["measurements"]["brightness_patterns"] is None


@pytest.mark.parametrize("bad_field,bad_value", [
    ("source_shape_yx", [10, 10]), ("source_region_level0_xyxy", [0, 0, 10, 10]),
    ("artifact_accuracy_validated", True), ("quality_decision", "accept"),
])
def test_inconsistent_brightness_evidence_is_rejected(tmp_path, bad_field, bad_value):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    source = source_dir / "source.tif"
    tifffile.imwrite(source, stripe_image(), photometric="minisblack")
    inspection = tmp_path / "inspection"
    session.create_session(source, inspection)
    record = measure_image(source, brightness_config=config(period_native_px=128))
    record["brightness_patterns"][bad_field] = bad_value
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Brightness measurement scope"):
        build_packet(inspection, record=path, expected_record_sha256=session.sha(path))
