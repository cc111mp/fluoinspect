"""Source-bound core-region proposals, original-pixel statistics and optional local evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import tifffile

from .core import CoreConfig, LABELS, measure_region_intensity, propose_core
from .analysis import attribute_candidates
from ..detectors.local import LocalConfig, measure_local_artifacts, validate_local_evidence
from ..investigation.session import artifact_path, fingerprint, native_hash, sha
from ..investigation.triage import review_triage
from ..io.native import read_native
from ..io.persistence import atomic_json


def _cached_local(path, expected, file_hash, pixel_hash, shape, modality):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Cached local evidence requires its independently retained SHA-256")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Cached local evidence hash disagrees")
    record = json.loads(data)
    if (record.get("schema") != "fluoinspect.local-scan.v1"
            or record.get("source_file_sha256") != file_hash
            or record.get("source_pixel_sha256") != pixel_hash
            or record.get("source_shape_yx") != list(shape)
            or record.get("modality") != modality
            or record.get("source_pixels_preserved") is not True
            or record.get("quality_decision") is not None):
        raise ValueError("Cached local evidence is not bound to this source")
    local = record["local_artifacts"]
    validate_local_evidence(local, shape, {"engine_id": local["engine_id"], "configuration": local["configuration"]})
    triage = record.get("review_triage")
    if not isinstance(triage, dict) or triage != review_triage(
            periodic_state=triage.get("periodic_screen_state"), local_artifacts=local):
        raise ValueError("Cached local review state disagrees")
    return local, {"path": str(Path(path).resolve()), "sha256": expected, "run_identity": record["run_identity"]}


def analyze_core(source, *, config=None, modality="autofluorescence", target_point_native_xy=None,
                 local_config=None, cached_local_scan=None, expected_local_scan_sha256=None,
                 expected_source_sha256=None):
    config = CoreConfig() if config is None else config
    if not isinstance(config, CoreConfig):
        raise ValueError("A CoreConfig is required")
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Invalid modality")
    if (local_config is not None and not isinstance(local_config, LocalConfig)
            or local_config is not None and cached_local_scan is not None
            or cached_local_scan is None and expected_local_scan_sha256 is not None):
        raise ValueError("Choose fresh local measurements or independently hashed cached evidence")
    source = Path(source).resolve()
    before = fingerprint(source)
    file_hash = sha(source)
    if expected_source_sha256 is not None and file_hash != expected_source_sha256:
        raise ValueError("Source file hash disagrees")
    native = read_native(source)
    native.flags.writeable = False
    pixels_hash = native_hash(native)
    proposal, labels = propose_core(native, config=config, target_point_native_xy=target_point_native_xy)
    local, lineage = None, None
    if cached_local_scan is not None:
        local, lineage = _cached_local(cached_local_scan, expected_local_scan_sha256,
                                       file_hash, pixels_hash, native.shape, modality)
    elif local_config is not None:
        local = measure_local_artifacts(native, config=local_config)
    attribution = attribute_candidates(local, proposal, labels) if local is not None else None
    stats = measure_region_intensity(native, labels, proposal["native_pixels_per_label_cell"])
    triage = review_triage(periodic_state="unverified", local_artifacts=local,
                           target_identity_status=proposal["target_identity_status"])
    triage["reasons"].append("core_boundary_not_reviewed")
    if fingerprint(source) != before or sha(source) != file_hash or native_hash(native) != pixels_hash:
        raise ValueError("Source changed during core-region analysis")
    identity = {"engine_id": proposal["engine_id"], "configuration": config.to_dict(),
                "source_file_sha256": file_hash, "source_pixel_sha256": pixels_hash, "modality": modality,
                "target_point_native_xy": proposal["target_point_native_xy"], "local_evidence": lineage,
                "local_configuration": local["configuration"] if local is not None else None}
    result = {"schema": "fluoinspect.core-regions.v1", "source": str(source), "modality": modality,
              "source_file_sha256": file_hash, "source_pixel_sha256": pixels_hash,
              "source_shape_yx": list(native.shape), "pixel_hash_byte_order": "little",
              "source_pixels_preserved": True, "proposal": proposal, "region_intensity": stats,
              "cached_local_lineage": lineage,
              "local_method": {"engine_id": local["engine_id"], "configuration": local["configuration"]}
              if local is not None else None,
              "local_candidate_attribution": attribution, "review_triage": triage,
              "run_identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "quality_decision": None}
    return result, labels


def write_core_result(destination, result, labels):
    destination = Path(destination)
    source = Path(result["source"]).resolve()
    if destination.resolve().is_relative_to(source.parent) or source.is_relative_to(destination.resolve()):
        raise ValueError("Output must remain outside the source image directory")
    destination.mkdir(parents=True, exist_ok=False)
    label_path = destination / "regions.tif"
    tifffile.imwrite(label_path, labels, photometric="minisblack")
    with tifffile.TiffFile(label_path) as tf:
        exported = tf.pages[0].asarray(maxworkers=1)
    if not np.array_equal(exported, labels):
        raise ValueError("Region labels differ on readback")
    result = dict(result)
    result["region_labels"] = {"path": "regions.tif", "file_sha256": sha(label_path),
                               "pixel_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                               "shape_yx": list(labels.shape), "dtype": "uint8", "exact_label_readback": True}
    atomic_json(destination / "core_regions.json", result)
    row = {"Target_identity_status": result["proposal"]["target_identity_status"],
           "Core_envelope_bbox_level0_xyxy": result["proposal"]["core_envelope_bbox_level0_xyxy"],
           "Proposal_count": len(result["proposal"]["proposals"]),
           "Unresolved_reasons": json.dumps(result["proposal"]["unresolved_reasons"]),
           "Verification_state": "unverified", "Expert_corrected_core_boundary": "",
           "Expert_core_identity_confirmed": "", "Quantitative_intensity_verdict": "",
           "Morphology_modelling_verdict": "", "Final_quality_decision": ""}
    with (destination / "core_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    attributed = result["local_candidate_attribution"]
    if attributed is not None:
        fields = ["candidate_id", "kind", "bbox_level0_xyxy", "scope", "core_envelope_fraction",
                  "core_context_fraction", "tissue_support_fraction", "nearby_background_fraction",
                  "neighbour_guard_fraction", "native_log_contrast", "target_identity_status", "quality_decision"]
        with (destination / "candidate_regions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: candidate[key] for key in fields} for candidate in attributed["candidates"])
    return result


def verify_core_result(path, *, expected_record_sha256):
    """Rebind source pixels, coarse masks and proposals; reject edited identity/QC claims."""
    path = Path(path).resolve()
    data = path.read_bytes()
    if expected_record_sha256 is None or hashlib.sha256(data).hexdigest() != expected_record_sha256:
        raise ValueError("Core record hash disagrees")
    result = json.loads(data)
    if (result.get("schema") != "fluoinspect.core-regions.v1" or result.get("quality_decision") is not None
            or result.get("modality") not in {"autofluorescence", "fluorescence"}
            or result.get("pixel_hash_byte_order") != "little"
            or result.get("source_pixels_preserved") is not True):
        raise ValueError("Invalid core record interpretation")
    source = Path(result["source"])
    before = fingerprint(source)
    if sha(source) != result["source_file_sha256"]:
        raise ValueError("Core source file changed")
    native = read_native(source)
    if native_hash(native) != result["source_pixel_sha256"] or list(native.shape) != result["source_shape_yx"]:
        raise ValueError("Core native source changed")
    metadata = result["region_labels"]
    label_path = artifact_path(path.parent, metadata["path"])
    if sha(label_path) != metadata["file_sha256"]:
        raise ValueError("Region label file changed")
    proposal = result["proposal"]
    expected_shape = [math.ceil(v / proposal["native_pixels_per_label_cell"]) for v in native.shape]
    with tifffile.TiffFile(label_path) as tf:
        if (len(tf.pages) != 1 or tf.pages[0].axes != "YX" or tf.pages[0].dtype != np.dtype("uint8")
                or list(tf.pages[0].shape) != expected_shape or math.prod(expected_shape) > 1024 ** 2):
            raise ValueError("Invalid bounded region-label TIFF")
        labels = tf.pages[0].asarray(maxworkers=1)
    if (metadata["shape_yx"] != expected_shape or metadata.get("dtype") != "uint8"
            or metadata.get("exact_label_readback") is not True
            or hashlib.sha256(labels.tobytes()).hexdigest() != metadata["pixel_sha256"]
            or not np.isin(labels, list(LABELS.values())).all()):
        raise ValueError("Region label pixels disagree")
    config = CoreConfig(**proposal["configuration"])
    point = None if proposal["selection_basis"] == "export_center_prior_hypothesis" else proposal["target_point_native_xy"]
    recomputed, recomputed_labels = propose_core(native, config=config, target_point_native_xy=point)
    if proposal != recomputed or not np.array_equal(labels, recomputed_labels):
        raise ValueError("Core proposal or masks disagree with original source geometry")
    local = None
    lineage = result.get("cached_local_lineage")
    if lineage is not None:
        local, checked = _cached_local(lineage["path"], lineage["sha256"], result["source_file_sha256"],
                                       result["source_pixel_sha256"], native.shape, result["modality"])
        if checked != lineage:
            raise ValueError("Cached local lineage disagrees")
    elif result.get("local_method") is not None:
        options = dict(result["local_method"]["configuration"])
        options["widths_native_px"] = tuple(options["widths_native_px"])
        local = measure_local_artifacts(native, config=LocalConfig(**options))
    expected_attribution = attribute_candidates(local, proposal, labels) if local is not None else None
    method = {"engine_id": local["engine_id"], "configuration": local["configuration"]} if local is not None else None
    if result.get("local_method") != method:
        raise ValueError("Local method provenance disagrees")
    identity = {"engine_id": proposal["engine_id"], "configuration": config.to_dict(),
                "source_file_sha256": result["source_file_sha256"], "source_pixel_sha256": result["source_pixel_sha256"],
                "modality": result["modality"], "target_point_native_xy": proposal["target_point_native_xy"],
                "local_evidence": lineage, "local_configuration": local["configuration"] if local is not None else None}
    if result.get("run_identity") != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest():
        raise ValueError("Core run identity disagrees")
    triage = review_triage(periodic_state="unverified", local_artifacts=local,
                           target_identity_status=proposal["target_identity_status"])
    triage["reasons"].append("core_boundary_not_reviewed")
    if (result.get("local_candidate_attribution") != expected_attribution or result.get("review_triage") != triage
            or result.get("region_intensity") != measure_region_intensity(native, labels, proposal["native_pixels_per_label_cell"])):
        raise ValueError("Core measurements or pending review policy disagree")
    if fingerprint(source) != before or sha(source) != result["source_file_sha256"]:
        raise ValueError("Source changed during core verification")
    return result, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    parser.add_argument("--target-point", nargs=2, type=float, help="Explicit native XY prior; remains unreviewed")
    parser.add_argument("--expected-source-sha256")
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--scan-local", action="store_true", help="Measure original unmasked pixels on the systematic detector grid")
    local.add_argument("--cached-local-scan", type=Path)
    parser.add_argument("--expected-local-scan-sha256")
    args = parser.parse_args()
    result, labels = analyze_core(args.source, modality=args.modality, target_point_native_xy=args.target_point,
                                 local_config=LocalConfig() if args.scan_local else None,
                                 cached_local_scan=args.cached_local_scan,
                                 expected_local_scan_sha256=args.expected_local_scan_sha256,
                                 expected_source_sha256=args.expected_source_sha256)
    result = write_core_result(args.output, result, labels)
    print(json.dumps({"record": str(args.output / "core_regions.json"),
                      "target_identity_status": result["proposal"]["target_identity_status"],
                      "verification_state": "unverified", "quality_decision": None}))
