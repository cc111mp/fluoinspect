import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.pipeline import measure_image
from fluoinspect.investigation.session import sha
from fluoinspect.investigation import session as investigation
from fluoinspect.tools.evidence import build_packet
from fluoinspect.detectors import axial


@pytest.mark.parametrize("modality", ["autofluorescence", "fluorescence"])
def test_measurements_preserve_signal_and_keep_quality_decisions_pending(tmp_path, modality):
    source = tmp_path / "source.tif"
    values = np.full((800, 800), 100, np.uint16)
    values[:, 400:] = 1000
    tifffile.imwrite(source, values, photometric="minisblack")
    before = sha(source)
    result = measure_image(source, modality=modality)
    assert result["modality"] == modality
    assert result["native_confirmed_patterns"] >= 1
    assert result["metrics"]["raw"]["mean"] == pytest.approx(550)
    assert result["source_pixels_preserved"] and sha(source) == before
    assert result["human_decision"] == "" and result["automated_assessment"] == "measurements_only"
    assert not result["central_proxy_assessed"]


def test_installed_cli_can_measure_without_workstation_paths(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "example.tif"
    tifffile.imwrite(source, np.full((80, 80), 500, np.uint16), photometric="minisblack")
    output = tmp_path / "output"
    process = subprocess.run([sys.executable, "-m", "fluoinspect", "measure", str(source),
                              "--modality", "fluorescence", "--output", str(output)],
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    result = json.loads((output / "measurements.json").read_text())
    assert result["modality"] == "fluorescence" and result["human_decision"] == ""


def test_fluorescence_modality_reaches_evidence_without_claiming_unexported_native_views(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "fluorescence.tif"
    tifffile.imwrite(source, np.full((80, 80), 500, np.uint16), photometric="minisblack")
    session = tmp_path / "session"
    investigation.create_session(source, session, modality="fluorescence")
    first = build_packet(session)
    assert first["source"]["modality"] == "fluorescence"
    assert first["verification"]["native_crop_count"] == 0
    assert not first["verification"]["native_crops_checked"]
    investigation.request_view(session, [10, 10, 40, 40], purpose="Comparison region", kind="control")
    second = build_packet(session)
    assert second["verification"]["native_crop_count"] == 1
    assert second["verification"]["native_crops_checked"] and second["human_QC_decision"] == ""


def test_same_pixels_with_inconsistent_modality_metadata_are_not_combined(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "fluorescence.tif"
    tifffile.imwrite(source, np.full((80, 80), 500, np.uint16), photometric="minisblack")
    session = tmp_path / "session"
    investigation.create_session(source, session, modality="fluorescence")
    record = measure_image(source, modality="autofluorescence")
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="modalities disagree"):
        build_packet(session, record=record_path, expected_record_sha256=sha(record_path))


def test_different_detector_settings_have_distinct_run_identities(tmp_path, monkeypatch):
    source = tmp_path / "example.tif"
    tifffile.imwrite(source, np.full((80, 80), 500, np.uint16), photometric="minisblack")
    first = measure_image(source)
    monkeypatch.setitem(axial.SETTINGS, "verify_min_jump", 0.08)
    second = measure_image(source)
    assert first["pixel_sha256"] == second["pixel_sha256"]
    assert first["run_identity"] != second["run_identity"]
    assert first["measurement_configuration"]["axial_detector"]["settings"]["verify_min_jump"] == 0.06
    assert second["measurement_configuration"]["axial_detector"]["settings"]["verify_min_jump"] == 0.08
