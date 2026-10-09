"""Portable single-plane measurement; no artifact acceptance is assigned."""
import argparse
import hashlib
import json
from pathlib import Path

from . import __version__
from .detectors import axial
from .investigation.session import fingerprint, native_hash, sha
from .io.native import read_native
from .io.persistence import atomic_json
from .measurements.supporting import compute_supporting_metrics


def measure_image(source, *, modality="autofluorescence"):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Modality must be autofluorescence or fluorescence")
    methods = {
        "package_version": __version__,
        "axial_detector": {"engine_id": axial.ENGINE_ID, "settings": dict(axial.SETTINGS),
                           "confirm_native_rows": axial.CONFIRM_ROWS,
                           "overview_block_size": axial.F},
        "supporting_metrics": {"patch_size": 1024, "overview_max_dim": 1024,
                               "background_cells_per_axis": 8},
    }
    methods_sha = hashlib.sha256(json.dumps(methods, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = Path(source).resolve()
    before = fingerprint(source)
    file_sha = sha(source)
    native = read_native(source)
    native.flags.writeable = False
    if fingerprint(source) != before:
        raise ValueError("Source changed during decode")
    pixels_sha = native_hash(native)
    _, _, candidates, _, _ = axial.screen_native(native)
    metrics = compute_supporting_metrics(native, **methods["supporting_metrics"])
    if fingerprint(source) != before or sha(source) != file_sha or native_hash(native) != pixels_sha:
        raise ValueError("Source file or decoded pixels changed during measurement")
    unique = [c for c in candidates if c["unique_verified_pattern"]]
    # The compatibility record permits existing evidence-packet readers. The
    # staging section explicitly records a direct read, not a verified copy.
    return {
        "schema": "af-qc.finish-image.v2", "producer": "fluoinspect", "status": "complete",
        "asset_id": hashlib.sha256((source.name + ":" + file_sha).encode()).hexdigest()[:16],
        "relative_path": source.name, "modality": modality,
        "run_identity": hashlib.sha256((file_sha + ":" + modality + ":" + methods_sha).encode()).hexdigest(),
        "measurement_configuration": methods, "measurement_configuration_sha256": methods_sha,
        "shape_yx": list(native.shape), "pixel_sha256": pixels_sha, "pixel_hash_byte_order": "little",
        "source_pixels_preserved": True,
        "staging": {"file_sha256": file_sha, "source_access": "direct local read; no staging copy claimed"},
        "native_confirmed_patterns": len(unique), "scored_lines": None,
        "central_proxy_assessed": False, "central_proxy_is_reviewed_core_boundary": False,
        "lines": candidates, "metrics": metrics,
        "automated_assessment": "measurements_only", "validation_status": "unvalidated_for_quality_decisions",
        "human_decision": "", "reviewed_issues": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory outside the input directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source.parent) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source image directory")
    result = measure_image(source, modality=args.modality)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "measurements.json", result)
    print(json.dumps({"record": str(output / "measurements.json"),
                      "modality": args.modality, "assessment": "measurements_only"}, indent=2))
