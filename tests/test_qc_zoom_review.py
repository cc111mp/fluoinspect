"""Coordinate, native-pixel, budget and evidence-contract tests on synthetic data."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.investigation import session as zoom


def sample(tmp_path, budget=None):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "sample_AF.tif"
    raw = (np.arange(161 * 173).reshape(161, 173) * 2 + 101).astype(np.uint16)
    tifffile.imwrite(source, raw, photometric="minisblack")
    checksum = zoom.sha(source)
    session = tmp_path / "session"
    overview = zoom.create_session(source, session, expected_sha256=checksum,
                                   display_high=60000, budget=budget)
    return source, raw, session, overview, checksum


def report(view_id="view-001", bbox=None):
    return {"schema": "af-qc.assistant-review.v1", "observation_status": "concerns_in_views",
            "findings": [{"type": "parallel_bands", "bbox_level0_xyxy": bbox or [10, 20, 60, 70],
                          "evidence_view_ids": [view_id], "observation": "Synthetic review observation.",
                          "cause_hypothesis": None, "certainty": "suspected"}],
            "unassessed_checks": ["Calibrated focus"], "limitations": ["Selected views only."]}


def test_partial_block_means_are_not_zero_padded():
    a = np.arange(35).reshape(5, 7).astype(np.uint16)
    actual = zoom.block_mean(a, 3)
    expected = np.array([[a[y:y+3, x:x+3].mean() for x in (0, 3, 6)] for y in (0, 3)])
    np.testing.assert_allclose(actual, expected)
    assert zoom.block_mean(a, 3)[-1, -1] == a[3:5, 6:7].mean()


def test_overview_source_geometry_and_blank_human_decision(tmp_path):
    source, raw, session, view, checksum = sample(tmp_path)
    assert view["bbox_level0_xyxy"] == [0, 0, 173, 161]
    assert view["preview_shape_yx"] == [161, 173]
    assert zoom.sha(source) == checksum
    info = json.loads((session / "session.json").read_text())
    assert info["human_QC_decision"] == "" and not info["source_images_modified"]
    assert view["raw_tiff"] is None
    np.testing.assert_array_equal(tifffile.imread(source), raw)


def test_crop_readback_and_shared_window(tmp_path):
    source, raw, session, overview, checksum = sample(tmp_path)
    view = zoom.request_view(session, [10, 20, 80, 90], purpose="Inspect a synthetic seam")
    cropped = tifffile.imread(zoom.artifact_path(session, view["raw_tiff"]))
    np.testing.assert_array_equal(cropped, raw[20:90, 10:80])
    assert cropped.dtype == np.uint16
    assert view["display_window"] == overview["display_window"]
    assert view["all_raw_crop_pixels_equal_source"] and zoom.sha(source) == checksum


def test_preview_mapping_keeps_clipped_edge_blocks(tmp_path):
    _, raw, session, overview, _ = sample(tmp_path, {"max_display_dimension": 64})
    assert overview["native_pixels_per_preview_pixel"] == 3
    assert overview["preview_shape_yx"] == [54, 58]
    view = zoom.request_view(session, [10, 11, 58, 54], purpose="Preview-selected detail",
                             coordinates="preview", parent_view_id="view-000")
    assert view["bbox_level0_xyxy"] == [30, 33, 173, 161]
    np.testing.assert_array_equal(tifffile.imread(zoom.artifact_path(session, view["raw_tiff"])), raw[33:161, 30:173])


@pytest.mark.parametrize("bbox", [[-1, 0, 10, 10], [0, 0, 174, 10], [10, 5, 10, 7], [True, 0, 10, 10]])
def test_bad_coordinates_are_errors(tmp_path, bbox):
    _, _, session, _, _ = sample(tmp_path)
    with pytest.raises(ValueError):
        zoom.request_view(session, bbox, purpose="Invalid requested region")
    assert len(json.loads((session / "session.json").read_text())["views"]) == 1


def test_changed_source_refuses_new_views(tmp_path):
    source, raw, session, _, _ = sample(tmp_path)
    tifffile.imwrite(source, raw + 1, photometric="minisblack")
    with pytest.raises(ValueError, match="Source changed"):
        zoom.request_view(session, [0, 0, 20, 20], purpose="Source must remain immutable")


def test_budget_prevents_unbounded_zoom(tmp_path):
    _, _, session, _, _ = sample(tmp_path, {"max_views": 2})
    zoom.request_view(session, [0, 0, 20, 20], purpose="One allowed detail")
    with pytest.raises(ValueError, match="budget exhausted"):
        zoom.request_view(session, [20, 20, 40, 40], purpose="Beyond budget")


def test_raw_crop_budget_still_allows_context(tmp_path):
    _, _, session, _, _ = sample(tmp_path, {"max_raw_crop_pixels": 100})
    with pytest.raises(ValueError, match="too large"):
        zoom.request_view(session, [0, 0, 40, 40], purpose="Too much native crop")
    view = zoom.request_view(session, [0, 0, 40, 40], purpose="Bounded context view", kind="context")
    assert view["raw_tiff"] is None


def test_model_cannot_assign_acceptance(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    value = report()
    value["observation_status"] = "accepted"
    with pytest.raises(ValueError, match="cannot assign image acceptance"):
        zoom.validate_report(session, value)


def test_evidence_must_cover_localized_finding(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    zoom.request_view(session, [10, 20, 80, 90], purpose="Evidence context", kind="context")
    with pytest.raises(ValueError, match="outside its evidence"):
        zoom.validate_report(session, report(bbox=[100, 100, 140, 140]))
    with pytest.raises(ValueError, match="known recorded evidence"):
        zoom.validate_report(session, report(view_id="invented-view"))


def test_valid_report_stays_unvalidated_and_preserves_human_fields(tmp_path):
    source, _, session, _, checksum = sample(tmp_path)
    zoom.request_view(session, [10, 20, 80, 90], purpose="Native evidence")
    value = zoom.validate_report(session, report())
    assert value["human_QC_decision"] == "" and not value["artifact_accuracy_validated"]
    assert value["report_geometry_and_evidence_validated"] and zoom.sha(source) == checksum
    with pytest.raises(FileExistsError):
        zoom.validate_report(session, report())


def test_tampered_evidence_refuses_report(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    view = zoom.request_view(session, [10, 20, 80, 90], purpose="Recorded evidence")
    zoom.artifact_path(session, view["png"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="preview changed"):
        zoom.validate_report(session, report())


def test_output_never_writes_inside_source_directory(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "source.tif"
    tifffile.imwrite(source, np.ones((32, 32), np.uint16), photometric="minisblack")
    with pytest.raises(ValueError, match="outside the source"):
        zoom.create_session(source, originals / "new-session")


def test_identified_retry_does_not_consume_budget(tmp_path):
    _, _, session, _, _ = sample(tmp_path, {"max_views": 2})
    args = {"purpose": "Retryable native evidence", "request_id": "native-001"}
    first = zoom.request_view(session, [10, 20, 80, 90], **args)
    second = zoom.request_view(session, [10, 20, 80, 90], **args)
    assert first == second
    assert zoom.inspect_session(session)["committed_views"] == 2
    with pytest.raises(ValueError, match="different arguments"):
        zoom.request_view(session, [11, 20, 80, 90], **args)


def test_another_process_cannot_modify_owned_session(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    with zoom.session_owner(session):
        process = subprocess.run(
            [sys.executable, "-m", "fluoinspect.investigation.session", "view", "--session", str(session),
             "--bbox", "10", "20", "80", "90", "--purpose", "Concurrent request"],
            capture_output=True, text=True, timeout=15)
    assert process.returncode != 0 and "Another controller owns" in process.stderr
    assert zoom.inspect_session(session)["committed_views"] == 1


def test_relative_evidence_survives_session_copy(tmp_path):
    _, raw, session, _, _ = sample(tmp_path)
    view = zoom.request_view(session, [10, 20, 80, 90], purpose="Portable evidence")
    assert not Path(view["png"]).is_absolute() and not Path(view["raw_tiff"]).is_absolute()
    copied = tmp_path / "copied-session"
    shutil.copytree(session, copied)
    shutil.rmtree(session)
    np.testing.assert_array_equal(tifffile.imread(zoom.artifact_path(copied, view["raw_tiff"])), raw[20:90, 10:80])
    assert zoom.inspect_session(copied)["committed_views"] == 2
    assert zoom.validate_report(copied, report())["human_QC_decision"] == ""


def test_interrupted_crop_is_not_overwritten_or_counted(tmp_path, monkeypatch):
    _, _, session, _, _ = sample(tmp_path)
    real_atomic = zoom.atomic_json

    def fail_ledger(path, value):
        if Path(path).name == "session.json":
            raise OSError("simulated interruption before ledger commit")
        return real_atomic(path, value)

    monkeypatch.setattr(zoom, "atomic_json", fail_ledger)
    with pytest.raises(OSError, match="simulated interruption"):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Interrupted evidence", request_id="retry-001")
    status = zoom.inspect_session(session)
    assert status["committed_views"] == 1 and len(status["unregistered_files"]) == 2
    orphan_hashes = {path: zoom.sha(session / path) for path in status["unregistered_files"]}
    monkeypatch.setattr(zoom, "atomic_json", real_atomic)
    view = zoom.request_view(session, [10, 20, 80, 90], purpose="Interrupted evidence", request_id="retry-001")
    assert view["png"] not in orphan_hashes
    assert all(zoom.sha(session / path) == checksum for path, checksum in orphan_hashes.items())
    assert zoom.inspect_session(session)["committed_views"] == 2


def test_source_change_during_decode_is_detected(tmp_path, monkeypatch):
    source, raw, session, _, _ = sample(tmp_path)
    original_read = zoom.read_native

    def changing_read(path):
        native = original_read(path)
        tifffile.imwrite(source, raw + 1, photometric="minisblack")
        return native

    monkeypatch.setattr(zoom, "read_native", changing_read)
    with pytest.raises(ValueError, match="during decode"):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Concurrent source change")


def test_decoded_pixel_mismatch_is_detected(tmp_path, monkeypatch):
    _, raw, session, _, _ = sample(tmp_path)
    monkeypatch.setattr(zoom, "read_native", lambda _: raw + 1)
    with pytest.raises(ValueError, match="Decoded source pixels"):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Decoder inconsistency")


def test_evidence_path_cannot_escape_session(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    info = json.loads((session / "session.json").read_text())
    info["views"][0]["png"] = "../../outside.png"
    zoom.atomic_json(session / "session.json", info)
    with pytest.raises(ValueError, match="escapes"):
        zoom.inspect_session(session)


def test_output_symlink_cannot_bypass_source_directory_guard(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "source.tif"
    tifffile.imwrite(source, np.ones((32, 32), np.uint16), photometric="minisblack")
    (tmp_path / "alias").symlink_to(originals, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the source"):
        zoom.create_session(source, tmp_path / "alias" / "new-session")


def test_sealed_session_refuses_new_views_and_report_tampering(tmp_path):
    _, _, session, _, _ = sample(tmp_path)
    zoom.request_view(session, [10, 20, 80, 90], purpose="Final evidence")
    zoom.validate_report(session, report())
    assert zoom.inspect_session(session)["state"] == "sealed"
    with pytest.raises(ValueError, match="sealed"):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Late mutation")
    (session / "assistant_review.json").write_text("{}")
    with pytest.raises(ValueError, match="report changed"):
        zoom.inspect_session(session)


def test_interrupted_report_is_preserved_for_explicit_recovery(tmp_path, monkeypatch):
    _, _, session, _, _ = sample(tmp_path)
    zoom.request_view(session, [10, 20, 80, 90], purpose="Final evidence")
    real_atomic = zoom.atomic_json

    def fail_seal(path, value):
        if Path(path).name == "session.json" and value.get("state") == "sealed":
            raise OSError("simulated interruption after report commit")
        return real_atomic(path, value)

    monkeypatch.setattr(zoom, "atomic_json", fail_seal)
    with pytest.raises(OSError, match="simulated interruption"):
        zoom.validate_report(session, report())
    status = zoom.inspect_session(session)
    assert status["state"] == "report_present_metadata_pending" and not status["resume_allowed"]
    with pytest.raises(ValueError, match="sealed"):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Must not overwrite evidence")


def test_status_can_audit_evidence_without_source_access(tmp_path):
    source, _, session, _, _ = sample(tmp_path)
    source.unlink()
    status = zoom.inspect_session(session)
    assert status["evidence_file_hashes_validated"] and not status["source_rechecked"]
    with pytest.raises(FileNotFoundError):
        zoom.request_view(session, [10, 20, 80, 90], purpose="Source unavailable")


def test_pixel_hash_has_explicit_portable_byte_order():
    raw = np.array([[1, 255, 256, 65535]], dtype="<u2")
    assert zoom.native_hash(raw) == zoom.native_hash(raw.astype(">u2"))
