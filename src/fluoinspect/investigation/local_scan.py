"""Source-bound local scanning of every scheduled tile; never auto-accepts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ..detectors.local import ENGINE_ID, LocalConfig, measure_local_artifacts
from ..io.native import read_native
from ..io.persistence import atomic_json
from .session import fingerprint, native_hash, sha
from .triage import review_triage


def scan_image(source, *, config=None, target_box=None, modality="autofluorescence",
               periodic_state="unassessed", expected_source_sha256=None):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Invalid modality")
    source = Path(source).resolve()
    config = LocalConfig() if config is None else config
    if not isinstance(config, LocalConfig):
        raise ValueError("A LocalConfig is required")
    before = fingerprint(source)
    file_sha = sha(source)
    if expected_source_sha256 is not None and file_sha != expected_source_sha256:
        raise ValueError("Source file hash disagrees")
    native = read_native(source)
    native.flags.writeable = False
    pixel_sha = native_hash(native)
    local = measure_local_artifacts(native, config=config, target_box=target_box)
    if fingerprint(source) != before or sha(source) != file_sha or native_hash(native) != pixel_sha:
        raise ValueError("Source file or decoded pixels changed during local scan")
    identity = {"engine_id": ENGINE_ID, "configuration": config.to_dict(),
                "target_region_level0_xyxy": local["target_region_level0_xyxy"],
                "modality": modality, "source_file_sha256": file_sha, "source_pixel_sha256": pixel_sha,
                "periodic_screen_state": periodic_state}
    triage = review_triage(periodic_state=periodic_state, local_artifacts=local)
    return {"schema": "fluoinspect.local-scan.v1", "source": str(source), "modality": modality,
            "source_file_sha256": file_sha, "source_pixel_sha256": pixel_sha,
            "pixel_hash_byte_order": "little", "source_shape_yx": list(native.shape),
            "run_identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "source_pixels_preserved": True, "local_artifacts": local, "review_triage": triage,
            "quality_decision": None}


def write_scan(destination, result):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "local_scan.json", result)
    fields = ["tile_id", "bbox_level0_xyxy", "status", "proposal_native_pixels_per_cell",
              "candidate_count", "proposals_omitted", "verification_state", "quality_decision"]
    with (destination / "tile_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for tile in result["local_artifacts"]["tiles"]:
            writer.writerow({**{key: tile[key] for key in fields[:6]},
                             "verification_state": "unverified", "quality_decision": ""})
    fields = ["verification_state", "review_required", "route", "periodic_screen_state",
              "local_candidate_count", "quality_decision", "quantitative_intensity", "morphology_modelling"]
    with (destination / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**{key: result["review_triage"][key] for key in fields[:5]},
                         **{key: "" for key in fields[5:]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    parser.add_argument("--bbox", nargs=4, type=int)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--max-tiles", type=int, default=512)
    parser.add_argument("--periodic-screen-state", choices=["flagged", "unflagged", "unassessed", "unverified"],
                        default="unassessed", help="Optional caller-supplied screen state; never a clean-quality finding")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if destination.is_relative_to(source.parent) or source.is_relative_to(destination):
        raise ValueError("Output must remain outside the source image directory")
    config = LocalConfig(tile_size=args.tile_size, stride=args.stride, max_tiles=args.max_tiles)
    result = scan_image(source, config=config, target_box=args.bbox, modality=args.modality,
                        periodic_state=args.periodic_screen_state, expected_source_sha256=args.expected_source_sha256)
    write_scan(destination, result)
    print(json.dumps({"record": str(destination / "local_scan.json"),
                      "candidates": result["local_artifacts"]["candidate_count"],
                      "detector_evaluated_area_fraction": result["local_artifacts"]["coverage"]["detector_evaluated_area_fraction"],
                      "verification_state": "unverified", "quality_decision": None}))
