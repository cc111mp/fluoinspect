"""Exact analysis crops with a checked mapping to their parent export.

A supplied rectangle is an unreviewed analysis envelope, not core segmentation.
All native values, including internal zero and dim pixels, are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from . import session as zoom
from ..io.persistence import atomic_json

SCHEMA = "fluoinspect.analysis-region.v1"


def prepare_region(source, destination, box, *, expected_source_sha256=None,
                   modality="autofluorescence", target_identity_status="unreviewed"):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Unsupported modality")
    if target_identity_status not in {"unreviewed", "unresolved"}:
        raise ValueError("Unsupported target identity status")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Region output already exists")
    if destination.is_relative_to(source.parent) or source.is_relative_to(destination):
        raise ValueError("Region output must remain outside the source directory")
    before = zoom.fingerprint(source)
    file_hash = zoom.sha(source)
    if expected_source_sha256 is not None and expected_source_sha256 != file_hash:
        raise ValueError("Source hash differs from the expected export")
    native = zoom.read_native(source)
    native.flags.writeable = False
    box = zoom.validate_box(box, native.shape)
    x0, y0, x1, y1 = box
    if min(x1 - x0, y1 - y0) < 8:
        raise ValueError("Analysis region must support at least one 8-pixel overview block")
    pixels_hash = zoom.native_hash(native)
    crop = native[y0:y1, x0:x1]
    destination.mkdir(parents=True)
    child = destination / "region.tif"
    tifffile.imwrite(child, crop, photometric="minisblack")
    exported = zoom.read_native(child)
    if not np.array_equal(exported, crop):
        raise ValueError("Analysis crop differs from the original pixels")
    if zoom.fingerprint(source) != before or zoom.sha(source) != file_hash:
        raise ValueError("Parent source changed during region preparation")
    receipt = {
        "schema": SCHEMA, "modality": modality,
        "parent_source": str(source), "parent_file_sha256": file_hash,
        "parent_pixel_sha256": pixels_hash, "parent_shape_yx": list(native.shape),
        "bbox_parent_level0_xyxy": box, "child_file": "region.tif",
        "child_file_sha256": zoom.sha(child), "child_pixel_sha256": zoom.native_hash(exported),
        "child_shape_yx": list(exported.shape), "pixel_hash_byte_order": "little",
        "coordinate_mapping": "parent XY = child XY + rectangle top-left; no resampling",
        "region_identity_reviewed": False, "core_boundary_validated": False,
        "target_identity_status": target_identity_status,
        "selection": "caller_supplied_unreviewed_rectangle_envelope",
        "dark_pixels_retained": True, "native_crop_exact_readback": True,
        "parent_source_modified": False, "quality_decision": None,
    }
    atomic_json(destination / "region.json", receipt)
    return receipt


def verify_region(receipt_path, *, expected_receipt_sha256):
    """Recheck both sources and the exact native mapping before using a cached crop."""
    if not isinstance(receipt_path, (str, Path)):
        raise ValueError("A region receipt path is required")
    receipt_path = Path(receipt_path).resolve()
    receipt_bytes = receipt_path.read_bytes()
    checksum = hashlib.sha256(receipt_bytes).hexdigest()
    if expected_receipt_sha256 is None or checksum != expected_receipt_sha256:
        raise ValueError("Region receipt differs from its independently retained hash")
    receipt = json.loads(receipt_bytes)
    if (receipt.get("schema") != SCHEMA or receipt.get("region_identity_reviewed") is not False
            or receipt.get("core_boundary_validated") is not False
            or receipt.get("dark_pixels_retained") is not True
            or receipt.get("native_crop_exact_readback") is not True
            or receipt.get("parent_source_modified") is not False
            or receipt.get("quality_decision") is not None
            or receipt.get("target_identity_status") not in {"unreviewed", "unresolved"}
            or receipt.get("modality") not in {"autofluorescence", "fluorescence"}):
        raise ValueError("Invalid analysis-region receipt")
    parent = Path(receipt["parent_source"]).resolve()
    child = zoom.artifact_path(receipt_path.parent, receipt["child_file"])
    parent_before, child_before = zoom.fingerprint(parent), zoom.fingerprint(child)
    if (zoom.sha(parent) != receipt["parent_file_sha256"]
            or zoom.sha(child) != receipt["child_file_sha256"]):
        raise ValueError("Region parent or child file changed")
    native, cropped = zoom.read_native(parent), zoom.read_native(child)
    if (list(native.shape) != receipt["parent_shape_yx"]
            or list(cropped.shape) != receipt["child_shape_yx"]
            or zoom.native_hash(native) != receipt["parent_pixel_sha256"]
            or zoom.native_hash(cropped) != receipt["child_pixel_sha256"]):
        raise ValueError("Region geometry or decoded pixels disagree")
    x0, y0, x1, y1 = zoom.validate_box(receipt["bbox_parent_level0_xyxy"], native.shape)
    if not np.array_equal(cropped, native[y0:y1, x0:x1]):
        raise ValueError("Region crop disagrees with its parent rectangle")
    if (zoom.fingerprint(parent) != parent_before or zoom.fingerprint(child) != child_before
            or zoom.sha(parent) != receipt["parent_file_sha256"]
            or zoom.sha(child) != receipt["child_file_sha256"]):
        raise ValueError("Region source changed during verification")
    return receipt, child


def map_box_to_parent(box, receipt):
    """Map child-local native coordinates to this receipt's parent export."""
    local = zoom.validate_box(box, receipt["child_shape_yx"])
    x0, y0, x1, y1 = zoom.validate_box(receipt["bbox_parent_level0_xyxy"], receipt["parent_shape_yx"])
    if receipt["child_shape_yx"] != [y1 - y0, x1 - x0]:
        raise ValueError("Region mapping geometry disagrees")
    return [local[0] + x0, local[1] + y0, local[2] + x0, local[3] + y0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bbox", required=True, nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--target-identity-status", choices=["unreviewed", "unresolved"], default="unreviewed")
    parser.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = prepare_region(args.source, args.output, args.bbox,
                             expected_source_sha256=args.expected_source_sha256, modality=args.modality,
                             target_identity_status=args.target_identity_status)
    print(json.dumps({"receipt": str(args.output / "region.json"),
                      "receipt_sha256": zoom.sha(args.output / "region.json"),
                      "child_source": str(args.output / "region.tif"),
                      "region_identity_reviewed": receipt["region_identity_reviewed"]}))
