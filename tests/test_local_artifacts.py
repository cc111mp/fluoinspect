"""Controlled local patterns, confounds, coverage and review-policy integration."""
import copy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.detectors.local import LocalConfig, measure_local_artifacts, validate_local_evidence
from fluoinspect.investigation import session
from fluoinspect.investigation.local_scan import scan_image
from fluoinspect.investigation.triage import review_triage
from fluoinspect.pipeline import measure_image
from fluoinspect.tools.evidence import build_packet


def cfg(**options):
    return LocalConfig(tile_size=512, stride=384, widths_native_px=(16, 64), **options)


def measure(values, **options):
    return measure_local_artifacts(values, config=cfg(**options))


def test_isolated_dark_hole_is_localized_and_exact_zeros_are_preserved():
    values = np.full((512, 512), 1000, np.uint16)
    values[200:280, 180:260] = 0
    before = values.copy()
    result = measure(values)
    candidates = [c for c in result["candidates"] if c["kind"] == "local_dark_region"]
    assert candidates
    assert any(c["center"]["zero_fraction"] == 1 for c in candidates)
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["bbox_level0_xyxy"]
        assert candidate["center"]["mean"] == pytest.approx(values[y0:y1, x0:x1].mean())
    assert np.array_equal(before, values) and result["quality_decision"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("transpose", [False, True])
def test_nonperiodic_step_has_native_boundary_and_flank_evidence(transpose):
    values = np.full((512, 512), 1000, np.uint16)
    values[:, 256:] = 250
    if transpose:
        values = values.T.copy()
    result = measure(values)
    candidates = [c for c in result["candidates"] if c["kind"] == "axial_intensity_step"]
    assert any(c["boundary_native_px"] == 256 and c["axis"] == ("y" if transpose else "x")
               and c["native_support_fraction"] == 1 for c in candidates)
    assert not result["artifact_accuracy_validated"]


def test_two_nonrepeating_lines_do_not_require_supported_periodic_cycles():
    values = np.full((512, 512), 1200, np.uint16)
    values[:, 160:176] = 0
    values[:, 352:368] = 0
    result = measure(values)
    boundaries = {c["boundary_native_px"] for c in result["candidates"] if c["kind"] == "axial_intensity_step"}
    assert boundaries.intersection({160, 176}) and boundaries.intersection({352, 368})


@pytest.mark.parametrize("kind", ["constant", "all_zero", "smooth_gradient", "low_noise"])
def test_controlled_smooth_signals_do_not_generate_abrupt_local_candidates(kind):
    if kind == "smooth_gradient":
        values = np.tile(np.linspace(500, 2000, 512).astype(np.uint16), (512, 1))
    elif kind == "low_noise":
        values = np.clip(1000 + np.random.default_rng(3).normal(0, 10, (512, 512)), 0, 65535).astype(np.uint16)
    else:
        values = np.full((512, 512), 0 if kind == "all_zero" else 1000, np.uint16)
    result = measure(values)
    assert result["candidate_count"] == 0
    assert review_triage(periodic_state="unflagged", local_artifacts=result)["verification_state"] == "unverified"


def test_natural_cavity_can_produce_same_observation_without_becoming_artifact_verdict():
    y, x = np.indices((512, 512))
    values = np.full((512, 512), 1000, np.uint16)
    values[(x - 260) ** 2 + (y - 230) ** 2 < 60 ** 2] = 30
    result = measure(values)
    assert result["candidate_count"] > 0
    assert all(not c["artifact_cause_verified"] and c["quality_decision"] is None for c in result["candidates"])


def test_candidate_in_unflagged_far_corner_is_evaluated_and_mapped():
    values = np.full((1301, 1403), 1000, np.uint16)
    values[1130:1210, 1210:1290] = 0
    result = measure(values)
    assert any(c["bbox_level0_xyxy"][0] > 1000 and c["bbox_level0_xyxy"][1] > 1000 for c in result["candidates"])
    assert result["coverage"]["complete_scheduled_grid_evaluated"]


@pytest.mark.parametrize("budget", [0, 1, 5, 512])
def test_coverage_matches_independent_native_raster_with_partial_edges_and_budget(budget):
    values = np.full((701, 803), 1000, np.uint16)
    target = [11, 17, 800, 698]
    result = measure_local_artifacts(values, config=cfg(max_tiles=budget), target_box=target)
    raster = np.zeros(values.shape, bool)
    for tile in result["tiles"]:
        x0, y0, x1, y1 = tile["bbox_level0_xyxy"]
        raster[y0:y1, x0:x1] = True
        assert tile["status"] == "detector_evaluated"
        assert tile["candidate_count"] == 0
    area = int(raster.sum())
    assert result["coverage"]["detector_evaluated_area_px"] == area
    assert result["coverage"]["detector_evaluated_area_fraction"] == pytest.approx(area / ((800 - 11) * (698 - 17)))
    assert result["coverage"]["model_inspected_area_fraction"] is None
    assert not result["coverage"]["detector_full_resolution_exhaustive"]


@pytest.mark.parametrize("state", ["flagged", "unflagged", "unassessed", "unverified"])
def test_all_periodic_states_keep_both_profile_decisions_pending(state):
    result = measure(np.full((512, 512), 1000, np.uint16))
    triage = review_triage(periodic_state=state, local_artifacts=result)
    assert triage["verification_state"] == "unverified" and triage["review_required"]
    assert triage["quality_decision"] is None
    assert all(value is None for value in triage["profile_decisions"].values())
    if state == "unflagged":
        assert "negative_periodic_screen_is_not_verified_clean" in triage["reasons"]


@pytest.mark.parametrize("options", [{"stride": 0}, {"max_tiles": -1}, {"max_tiles": True},
                                      {"min_log_contrast": float("nan")}, {"widths_native_px": (0,)},
                                      {"tile_size": 4096}, {"min_step_support_fraction": 2}])
def test_invalid_local_settings_are_rejected(options):
    with pytest.raises(ValueError):
        LocalConfig(**options)


def test_scan_verifies_source_hash_and_negative_cli_exports_unflagged_tiles(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "image.tif"
    tifffile.imwrite(source, np.full((512, 512), 1000, np.uint16), photometric="minisblack")
    expected = session.sha(source)
    with pytest.raises(ValueError, match="hash"):
        scan_image(source, expected_source_sha256="0" * 64)
    destination = tmp_path / "scan"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "scan", str(source), "--output", str(destination),
                              "--periodic-screen-state", "unflagged", "--expected-source-sha256", expected],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads((destination / "local_scan.json").read_text())
    assert result["source_file_sha256"] == expected == session.sha(source)
    assert result["source_pixels_preserved"] and result["review_triage"]["verification_state"] == "unverified"
    rows = list(csv.DictReader((destination / "tile_review.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 1 and rows[0]["candidate_count"] == "0" and rows[0]["quality_decision"] == ""


@pytest.mark.parametrize("mutation", ["area", "tile", "decision", "candidate_box", "triage"])
def test_evidence_binding_rejects_wrong_local_coverage_coordinates_and_acceptance(tmp_path, mutation):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "source.tif"
    values = np.full((512, 512), 1000, np.uint16)
    values[200:280, 180:260] = 0
    tifffile.imwrite(source, values, photometric="minisblack")
    record = measure_image(source, local_config=cfg())
    view_session = tmp_path / "session"
    session.create_session(source, view_session)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    packet = build_packet(view_session, record=path, expected_record_sha256=session.sha(path))
    assert packet["measurements"]["local_artifacts"]["candidate_count"] > 0
    assert packet["measurements"]["review_triage"]["verification_state"] == "unverified"
    changed = copy.deepcopy(record)
    local = changed["local_artifacts"]
    if mutation == "area":
        local["coverage"]["detector_evaluated_area_px"] -= 1
    elif mutation == "tile":
        local["tiles"][0]["status"] = "planned"
    elif mutation == "decision":
        local["quality_decision"] = "accepted"
    elif mutation == "candidate_box":
        local["candidates"][0]["bbox_level0_xyxy"] = [0, 0, 600, 600]
    else:
        changed["review_triage"]["quality_decision"] = "accepted"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        build_packet(view_session, record=path, expected_record_sha256=session.sha(path))


def test_json_roundtrip_and_default_measure_record_policy(tmp_path):
    result = measure(np.full((512, 512), 1000, np.uint16), max_tiles=0)
    loaded = json.loads(json.dumps(result))
    validate_local_evidence(loaded, [512, 512], {"engine_id": loaded["engine_id"], "configuration": loaded["configuration"]})
    triage = review_triage(periodic_state="unflagged", local_artifacts=result, target_identity_status="unresolved")
    assert "detector_coverage_incomplete" in triage["reasons"]
    assert "target_core_identity_unresolved" in triage["reasons"]
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.full((80, 80), 1000, np.uint16), photometric="minisblack")
    record = measure_image(source)
    assert record["local_artifacts"] is None and record["review_triage"]["verification_state"] == "unverified"
    assert "local_artifact_checks_not_run" in record["review_triage"]["reasons"]
