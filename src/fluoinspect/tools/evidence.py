"""Prepare verified AF views and measurements for a future model adapter.

Inspired by HarnessIR's evidence collection and separate verification roles.
This tool performs no model calls, image restoration or artifact classification.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..investigation import session as zoom
from ..io.persistence import atomic_json

from ..agents.prompts import ROLE_BRIEFS

from .registry import TOOL_SCOPES


def _bind_record(path, expected_sha256, info, pixel_sha256):
    if expected_sha256 is None:
        raise ValueError("A cached record requires its independently recorded expected SHA-256")
    path = Path(path)
    checksum = zoom.sha(path)
    if checksum != expected_sha256:
        raise ValueError("Cached record hash differs from the expected published record")
    record = json.loads(path.read_text())
    if (record.get("schema") != "af-qc.finish-image.v2" or record.get("status") != "complete"
            or record.get("source_pixels_preserved") is not True
            or record.get("shape_yx") != info["shape_yx"]
            or record.get("pixel_sha256") != pixel_sha256
            or record.get("staging", {}).get("file_sha256") != info["source_file_sha256"]):
        raise ValueError("Cached record is not verified evidence for this source")
    if record.get("modality") and info.get("modality") and record["modality"] != info["modality"]:
        raise ValueError("Measurement and investigation modalities disagree")
    metrics = record["metrics"]
    if metrics.get("source_shape_yx") != info["shape_yx"]:
        raise ValueError("Metric geometry disagrees with the source")
    return {
        "record_sha256": checksum, "asset_id": record["asset_id"], "run_identity": record["run_identity"],
        "measurement_configuration": record.get("measurement_configuration"),
        "measurement_configuration_sha256": record.get("measurement_configuration_sha256"),
        "interpretation": "measurements and screening candidates; no AF quality decision",
        "sharp_axial_patterns_anywhere": record["native_confirmed_patterns"],
        "sharp_axial_patterns_in_unreviewed_central_proxy": record["scored_lines"],
        "largest_same_position_support": max((v["exact_rows"] for v in record["lines"]), default=0),
        "raw": metrics["raw"], "relative_detail": metrics["detail_summary"],
        "background_summary": {k: v for k, v in metrics["background"].items() if k != "cells"},
        "caveats": metrics["caveats"], "artifact_detection_accuracy_validated": False,
    }


def build_packet(session, *, record=None, expected_record_sha256=None):
    """Audit source and committed views; bind optional measurements to the same source."""
    with zoom.session_owner(session):
        session, info, native = zoom._load(session)
        zoom.validate_artifacts(session, info)
        pixel_sha256 = zoom.native_hash(native)
        ids = [v["view_id"] for v in info["views"]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Recorded view IDs must be nonempty and unique")
        views = []
        for view in info["views"]:
            box = zoom.validate_box(view["bbox_level0_xyxy"], native.shape)
            x0, y0, x1, y1 = box
            step = view["native_pixels_per_preview_pixel"]
            if type(step) is not int or step < 1:
                raise ValueError("Invalid preview sampling scale")
            expected_shape = [math.ceil((y1 - y0) / step), math.ceil((x1 - x0) / step)]
            if view["preview_shape_yx"] != expected_shape:
                raise ValueError("Preview geometry disagrees with source mapping")
            png = zoom.artifact_path(session, view["png"])
            with Image.open(png) as preview:
                if preview.mode != "L" or [preview.height, preview.width] != expected_shape:
                    raise ValueError("Preview image dimensions or mode disagree with the receipt")
            raw_path = zoom.artifact_path(session, view["raw_tiff"]) if view["raw_tiff"] else None
            if raw_path is not None:
                crop = zoom.read_native(raw_path)
                if not np.array_equal(crop, native[y0:y1, x0:x1]):
                    raise ValueError("Recorded native crop disagrees with its source rectangle")
            views.append({
                "view_id": view["view_id"], "kind": view["kind"], "purpose": view["purpose"],
                "bbox_level0_xyxy": box, "preview_shape_yx": view["preview_shape_yx"],
                "native_pixels_per_preview_pixel": view["native_pixels_per_preview_pixel"],
                "display_window": view["display_window"],
                "png": str(png), "png_sha256": view["png_sha256"],
                "native_tiff": str(raw_path) if raw_path else None,
                "native_tiff_sha256": view["raw_tiff_sha256"],
                "native_crop_exact_readback": True if raw_path else None,
                "model_inspection_logged": False,
            })
        measurements = (_bind_record(record, expected_record_sha256, info, pixel_sha256)
                        if record is not None else None)
        if zoom.fingerprint(info["source"]) != info["source_fingerprint"]:
            raise ValueError("Source changed during packet preparation")
        sealed = info.get("state") == "sealed" or (session / "assistant_review.json").exists()
        return {
            "schema": "af-qc.evidence-packet.v1", "label": info["label"],
            "source": {"file_sha256": info["source_file_sha256"], "pixel_sha256": pixel_sha256,
                       "pixel_hash_byte_order": "little", "shape_yx": info["shape_yx"],
                       "coordinate_system": info["coordinate_system"], "modality": info.get("modality", "unspecified")},
            "views": views, "measurements": measurements,
            "tool_scopes": copy.deepcopy(TOOL_SCOPES), "fixed_role_briefs": dict(ROLE_BRIEFS),
            "next_crop_requests_allowed": not sealed,
            "session_state": "sealed_or_report_present" if sealed else "active",
            "verification": {"source_file_and_pixels_checked": True,
                             "committed_view_hashes_checked": True,
                             "native_crops_checked": any(v["native_tiff"] for v in views),
                             "native_crop_count": sum(v["native_tiff"] is not None for v in views),
                             "artifact_detection_accuracy_validated": False},
            "coverage": {"view_count": len(views), "full_resolution_image_inspection_established": False,
                         "note": "Exported crops and their resolution are logged; model inspection is not established."},
            "unassessed_checks": ["Calibrated artifact detection", "Geometric acquisition-overlap verification",
                                  "Before/after background correction", "Calibrated field homogeneity and focus"],
            "model_adapter_implemented": False, "model_inference_executed": False,
            "source_images_modified": False, "human_QC_decision": "",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--expected-record-sha256")
    parser.add_argument("--output", required=True, type=Path, help="Fresh packet directory outside source and session")
    args = parser.parse_args()
    destination, session = args.output.resolve(), args.session.resolve()
    info = json.loads((session / "session.json").read_text())
    source_parent = Path(info["source"]).resolve().parent
    if destination.is_relative_to(session) or destination.is_relative_to(source_parent):
        raise ValueError("Packet output must remain outside source and existing session directories")
    packet = build_packet(session, record=args.record, expected_record_sha256=args.expected_record_sha256)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "evidence_packet.json", packet)
    print(json.dumps({"packet": str(destination / "evidence_packet.json"), "views": len(packet["views"]),
                      "model_inference_executed": False, "human_QC_decision": ""}, indent=2))


if __name__ == "__main__":
    main()
