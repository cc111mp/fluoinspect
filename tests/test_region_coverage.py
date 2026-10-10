"""Source lineage, dark-region retention and independent geometric coverage checks."""
import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.investigation import session
from fluoinspect.investigation.coverage import _union_area, plan_tiles, summarize_views
from fluoinspect.investigation.regions import map_box_to_parent, prepare_region, verify_region
from fluoinspect.measurements.brightness import BrightnessConfig, measure_brightness_patterns
from fluoinspect.pipeline import measure_image
from fluoinspect.tools.evidence import build_packet


def fixture(tmp_path, modality="autofluorescence"):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "export.tif"
    pixels = np.arange(128 * 192, dtype=np.uint16).reshape(128, 192)
    pixels[40:70, 50:80] = 0
    tifffile.imwrite(source, pixels, photometric="minisblack")
    destination = tmp_path / "region"
    receipt = prepare_region(source, destination, [24, 16, 152, 112], modality=modality)
    return source, pixels, destination, receipt


def test_prepared_region_retains_dark_pixels_and_maps_exactly_to_parent(tmp_path):
    source, pixels, destination, receipt = fixture(tmp_path)
    before = session.sha(source)
    checked, child = verify_region(destination / "region.json",
                                   expected_receipt_sha256=session.sha(destination / "region.json"))
    cropped = session.read_native(child)
    assert np.array_equal(cropped, pixels[16:112, 24:152])
    assert int((cropped == 0).sum()) == 900
    assert map_box_to_parent([26, 24, 56, 54], checked) == [50, 40, 80, 70]
    assert receipt["child_shape_yx"] == [96, 128]
    assert receipt["dark_pixels_retained"] and not receipt["region_identity_reviewed"]
    assert not receipt["core_boundary_validated"] and receipt["quality_decision"] is None
    assert session.sha(source) == before


@pytest.mark.parametrize("which", ["parent", "child", "receipt_hash", "rectangle"])
def test_changed_region_sources_and_mappings_are_rejected(tmp_path, which):
    source, pixels, destination, receipt = fixture(tmp_path)
    path = destination / "region.json"
    expected = session.sha(path)
    if which == "parent":
        pixels[0, 0] += 1
        tifffile.imwrite(source, pixels, photometric="minisblack")
    elif which == "child":
        child = session.read_native(destination / "region.tif")
        child[10, 10] += 1
        tifffile.imwrite(destination / "region.tif", child, photometric="minisblack")
    else:
        receipt["bbox_parent_level0_xyxy"] = [25, 16, 153, 112]
        path.write_text(json.dumps(receipt))
        if which == "rectangle":
            expected = session.sha(path)
    with pytest.raises(ValueError):
        verify_region(path, expected_receipt_sha256=expected)


def test_receipt_is_parsed_from_the_same_bytes_that_were_hashed(tmp_path, monkeypatch):
    _, _, destination, _ = fixture(tmp_path)
    path = destination / "region.json"
    expected = session.sha(path)
    # A second read of mutable metadata would break the retained-hash contract.
    monkeypatch.setattr(type(path), "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("second metadata read")))
    receipt, _ = verify_region(path, expected_receipt_sha256=expected)
    assert receipt["native_crop_exact_readback"]


@pytest.mark.parametrize("modality", ["autofluorescence", "fluorescence"])
def test_region_lineage_reaches_measurements_and_packet_without_quality_verdict(tmp_path, modality):
    source, _, destination, receipt = fixture(tmp_path, modality)
    child = destination / "region.tif"
    expected = session.sha(destination / "region.json")
    result = measure_image(child, modality=modality, region_receipt=destination / "region.json",
                           expected_region_receipt_sha256=expected,
                           brightness_config=BrightnessConfig(foreground_method="region_envelope", erosion_cells=0))
    assert result["brightness_patterns"]["eligible_overview_fraction"] == 1
    assert result["source_region_lineage"]["bbox_parent_level0_xyxy"] == [24, 16, 152, 112]
    assert result["metrics"]["raw"]["zero_fraction"] > 0
    record = tmp_path / "record.json"
    record.write_text(json.dumps(result))
    inspection = tmp_path / "inspection"
    session.create_session(child, inspection, modality=modality)
    packet = build_packet(inspection, record=record, expected_record_sha256=session.sha(record))
    assert packet["measurements"]["source_region_lineage"]["parent_file_sha256"] == session.sha(source)
    assert not packet["coverage"]["full_resolution_image_inspection_established"]
    assert packet["human_QC_decision"] == "" and not receipt["region_identity_reviewed"]
    result["source_region_lineage"]["bbox_parent_level0_xyxy"][0] += 1
    record.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="lineage disagrees"):
        build_packet(inspection, record=record, expected_record_sha256=session.sha(record))


def test_region_requires_independent_receipt_hash_and_matching_child(tmp_path):
    source, _, destination, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="retained hash"):
        measure_image(destination / "region.tif", region_receipt=destination / "region.json")
    with pytest.raises(ValueError, match="source or modality"):
        measure_image(source, region_receipt=destination / "region.json",
                      expected_region_receipt_sha256=session.sha(destination / "region.json"))


def test_envelope_retains_zero_and_dim_intensities_without_brightness_exclusion():
    pixels = np.full((128, 256), 3000, np.uint16)
    pixels[:, :32] = 0
    pixels[:, 32:64] = 10
    result = measure_brightness_patterns(pixels, config=BrightnessConfig(foreground_method="region_envelope", erosion_cells=0))
    broad = result["axes"]["x"]["broad"]
    assert result["eligible_overview_fraction"] == 1
    assert broad["supported_samples"] == 256
    assert broad["p90_minus_p10_log1p"] > 7
    assert result["quality_decision"] is None and not result["region_mask_reviewed"]
    with pytest.raises(ValueError, match="zero erosion"):
        BrightnessConfig(foreground_method="region_envelope")


def test_union_area_matches_an_independent_pixel_raster():
    rng = np.random.default_rng(7)
    for _ in range(12):
        raster = np.zeros((29, 37), bool)
        boxes = []
        for _ in range(16):
            x0, x1 = sorted(rng.choice(38, size=2, replace=False).tolist())
            y0, y1 = sorted(rng.choice(30, size=2, replace=False).tolist())
            boxes.append([x0, y0, x1, y1])
            raster[y0:y1, x0:x1] = True
        assert _union_area(boxes) == int(raster.sum())


def test_systematic_grid_covers_partial_edges_and_records_bounded_omissions():
    complete = plan_tiles([1000, 1300], target_box=[30, 70, 1200, 900], tile_size=256, stride=192, max_views=64)
    assert complete["complete_grid_planned"]
    assert complete["planned_native_area_fraction"] == 1
    for view in complete["views"]:
        x0, y0, x1, y1 = view["bbox_level0_xyxy"]
        assert 30 <= x0 < x1 <= 1200 and 70 <= y0 < y1 <= 900
    limited = plan_tiles([1000, 1300], tile_size=256, stride=192, max_views=3)
    assert limited["view_budget_truncated"] and limited["omitted_grid_regions"] > 0
    assert limited["planned_native_area_fraction"] < 1
    assert len(limited["views"]) == 3 and not limited["model_inspection_established"]
    empty = plan_tiles([1000, 1300], max_views=0)
    assert empty["planned_native_area_fraction"] == 0 and not empty["complete_grid_planned"]


@pytest.mark.parametrize("options", [{"stride": 2048}, {"tile_size": True}, {"max_views": 65},
                                    {"target_box": [0, 0, 11, 20]}])
def test_invalid_coverage_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        plan_tiles([10, 10], **options)


def test_exported_coverage_unions_overlap_clips_scope_and_preserves_unknown_inspection():
    views = [{"bbox_level0_xyxy": [0, 0, 100, 100], "native_pixels_per_preview_pixel": 4, "native_tiff": None},
             {"bbox_level0_xyxy": [0, 0, 60, 100], "native_pixels_per_preview_pixel": 1, "native_tiff": "a.tif"},
             {"bbox_level0_xyxy": [40, 0, 100, 100], "native_pixels_per_preview_pixel": 1, "native_tiff": "b.tif"}]
    summary = summarize_views([100, 100], views, target_box=[10, 10, 90, 90])
    assert summary["native_exported_area_px"] == 6400
    assert summary["native_resolution_preview_area_fraction"] == 1
    assert summary["model_inspected_area_fraction"] is None
    assert not summary["full_resolution_image_inspection_established"]
    overview_only = summarize_views([100, 100], views[:1])
    assert overview_only["native_resolution_preview_area_px"] == 0
    assert overview_only["native_exported_area_px"] == 0


def test_coverage_cli_exports_source_bound_tiles_and_packet_accounts_for_native_area(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "example.tif"
    pixels = np.arange(160 * 160, dtype=np.uint16).reshape(160, 160)
    tifffile.imwrite(source, pixels, photometric="minisblack")
    inspection = tmp_path / "inspection"
    session.create_session(source, inspection, budget={"max_views": 16})
    output = tmp_path / "coverage"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "coverage", "--session", str(inspection),
                              "--tile-size", "80", "--stride", "64", "--export-views", "--output", str(output)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    plan = json.loads((output / "coverage_plan.json").read_text())
    assert len(plan["views"]) == 9 and all(v["status"] == "exported" for v in plan["views"])
    packet = build_packet(inspection)
    assert packet["coverage"]["native_exported_area_fraction"] == 1
    assert not packet["model_inference_executed"] and packet["human_QC_decision"] == ""


def test_region_and_envelope_measurement_cli(tmp_path):
    source, _, _, _ = fixture(tmp_path)
    output = tmp_path / "cli_region"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "region", "--source", str(source),
                              "--bbox", "24", "16", "152", "112", "--output", str(output)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    receipt = output / "region.json"
    measured = tmp_path / "measurements"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "measure", str(output / "region.tif"),
                              "--region-receipt", str(receipt), "--expected-region-receipt-sha256", session.sha(receipt),
                              "--pattern-foreground-method", "region_envelope", "--output", str(measured)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads((measured / "measurements.json").read_text())
    assert result["brightness_patterns"]["eligible_overview_fraction"] == 1
    assert result["source_region_lineage"]["parent_crop_mapping_verified"]
