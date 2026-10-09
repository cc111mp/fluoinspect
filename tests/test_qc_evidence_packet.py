"""Evidence binding and scope tests; no AF accuracy or model inference claims."""
import copy
import json

import numpy as np
import pytest
import tifffile

from fluoinspect.tools import evidence as packet
from fluoinspect.investigation import session as zoom
from fluoinspect.measurements.supporting import compute_supporting_metrics


def case(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "AF.tif"
    pixels = np.arange(160 * 170, dtype=np.uint16).reshape(160, 170)
    tifffile.imwrite(source, pixels, photometric="minisblack")
    session = tmp_path / "session"
    zoom.create_session(source, session, display_high=30000)
    view = zoom.request_view(session, [10, 20, 80, 90], purpose="Native evidence", kind="control")
    record = {"schema": "af-qc.finish-image.v2", "status": "complete", "asset_id": "synthetic-1",
              "run_identity": "synthetic-run", "shape_yx": list(pixels.shape), "source_pixels_preserved": True,
              "pixel_sha256": zoom.native_hash(pixels), "staging": {"file_sha256": zoom.sha(source)},
              "native_confirmed_patterns": 0, "scored_lines": 0, "lines": [],
              "metrics": compute_supporting_metrics(pixels)}
    record_path = tmp_path / "record.json"
    zoom.atomic_json(record_path, record)
    return source, pixels, session, view, record_path, record


def test_packet_preserves_source_and_cannot_promote_negative_score_to_acceptance(tmp_path):
    source, _, session, _, record_path, _ = case(tmp_path)
    before = zoom.sha(source)
    result = packet.build_packet(session, record=record_path, expected_record_sha256=zoom.sha(record_path))
    assert result["measurements"]["sharp_axial_patterns_anywhere"] == 0
    assert result["human_QC_decision"] == "" and not result["model_inference_executed"]
    assert not result["verification"]["artifact_detection_accuracy_validated"]
    assert not result["coverage"]["full_resolution_image_inspection_established"]
    assert all(not v["model_inspection_logged"] for v in result["views"])
    assert result["views"][1]["native_crop_exact_readback"]
    assert result["views"][1]["kind"] == "control"
    assert zoom.sha(source) == before


def test_absent_measurements_remain_absent(tmp_path):
    _, _, session, _, _, _ = case(tmp_path)
    result = packet.build_packet(session)
    assert result["measurements"] is None and result["next_crop_requests_allowed"]
    assert not result["tool_scopes"]["broad_oblique_band_detector"]["implemented"]


def test_record_requires_independent_hash_and_rejects_modification(tmp_path):
    _, _, session, _, record_path, record = case(tmp_path)
    with pytest.raises(ValueError, match="independently recorded"):
        packet.build_packet(session, record=record_path)
    expected = zoom.sha(record_path)
    record["scored_lines"] = 999
    zoom.atomic_json(record_path, record)
    with pytest.raises(ValueError, match="record hash differs"):
        packet.build_packet(session, record=record_path, expected_record_sha256=expected)


@pytest.mark.parametrize("field,value", [("shape_yx", [170, 160]), ("pixel_sha256", "f" * 64),
                                       ("source_pixels_preserved", False)])
def test_wrong_source_record_is_rejected_even_if_record_hash_matches(tmp_path, field, value):
    _, _, session, _, record_path, record = case(tmp_path)
    record[field] = value
    zoom.atomic_json(record_path, record)
    with pytest.raises(ValueError, match="not verified evidence"):
        packet.build_packet(session, record=record_path, expected_record_sha256=zoom.sha(record_path))


def test_crop_pixel_mismatch_is_rejected_even_if_ledger_hash_was_updated(tmp_path):
    _, pixels, session, view, _, _ = case(tmp_path)
    raw_path = zoom.artifact_path(session, view["raw_tiff"])
    tifffile.imwrite(raw_path, pixels[20:90, 10:80] + 1, photometric="minisblack")
    info = json.loads((session / "session.json").read_text())
    info["views"][1]["raw_tiff_sha256"] = zoom.sha(raw_path)
    zoom.atomic_json(session / "session.json", info)
    with pytest.raises(ValueError, match="source rectangle"):
        packet.build_packet(session)


def test_sealed_session_is_auditable_without_reopening_crop_requests(tmp_path):
    _, _, session, _, _, _ = case(tmp_path)
    report = {"schema": "af-qc.assistant-review.v1", "observation_status": "incomplete", "findings": [],
              "unassessed_checks": ["AF artifact accuracy"], "limitations": ["No model inference."]}
    zoom.validate_report(session, report)
    result = packet.build_packet(session)
    assert not result["next_crop_requests_allowed"] and result["human_QC_decision"] == ""


def test_duplicate_views_are_rejected(tmp_path):
    _, _, session, _, _, _ = case(tmp_path)
    info = json.loads((session / "session.json").read_text())
    info["views"].append(copy.deepcopy(info["views"][0]))
    zoom.atomic_json(session / "session.json", info)
    with pytest.raises(ValueError, match="unique"):
        packet.build_packet(session)


def test_incorrect_preview_source_mapping_is_rejected(tmp_path):
    _, _, session, _, _, _ = case(tmp_path)
    info = json.loads((session / "session.json").read_text())
    info["views"][0]["native_pixels_per_preview_pixel"] = 2
    zoom.atomic_json(session / "session.json", info)
    with pytest.raises(ValueError, match="Preview geometry"):
        packet.build_packet(session)
