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
from .measurements.brightness import BrightnessConfig, ENGINE_ID as BRIGHTNESS_ENGINE, measure_brightness_patterns


def measure_image(source, *, modality="autofluorescence", brightness_config=None):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Modality must be autofluorescence or fluorescence")
    brightness_config = BrightnessConfig() if brightness_config is None else brightness_config
    if not isinstance(brightness_config, BrightnessConfig):
        raise ValueError("A BrightnessConfig is required")
    methods = {
        "package_version": __version__,
        "axial_detector": {"engine_id": axial.ENGINE_ID, "settings": dict(axial.SETTINGS),
                           "confirm_native_rows": axial.CONFIRM_ROWS,
                           "overview_block_size": axial.F},
        "supporting_metrics": {"patch_size": 1024, "overview_max_dim": 1024,
                               "background_cells_per_axis": 8},
        "brightness_patterns": {"engine_id": BRIGHTNESS_ENGINE, "configuration": brightness_config.to_dict()},
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
    brightness = measure_brightness_patterns(native, config=brightness_config)
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
        "lines": candidates, "metrics": metrics, "brightness_patterns": brightness,
        "automated_assessment": "measurements_only", "validation_status": "unvalidated_for_quality_decisions",
        "human_decision": "", "reviewed_issues": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory outside the input directory")
    parser.add_argument("--pattern-period-px", type=float,
                        help="Explicit native-pixel period for experimental repetition measurements; absent by default")
    parser.add_argument("--pattern-period-basis", choices=["hypothesis", "acquisition_metadata"], default="hypothesis",
                        help="Provenance of the supplied period; neither option establishes artifact accuracy")
    parser.add_argument("--pattern-foreground-method", choices=["intensity_otsu", "hysteresis"], default="intensity_otsu",
                        help="Explicit provisional brightness-region selection; neither method establishes core identity")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source.parent) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source image directory")
    config = BrightnessConfig(period_native_px=args.pattern_period_px, period_basis=args.pattern_period_basis,
                              foreground_method=args.pattern_foreground_method)
    result = measure_image(source, modality=args.modality, brightness_config=config)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "measurements.json", result)
    print(json.dumps({"record": str(output / "measurements.json"),
                      "modality": args.modality, "assessment": "measurements_only"}, indent=2))
