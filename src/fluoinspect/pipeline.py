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
from .detectors.bands import BandConfig, ENGINE_ID as BANDS_ENGINE, measure_alternating_bands
from .investigation.regions import verify_region


def measure_image(source, *, modality="autofluorescence", brightness_config=None, band_config=None,
                  region_receipt=None, expected_region_receipt_sha256=None):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Modality must be autofluorescence or fluorescence")
    brightness_config = BrightnessConfig() if brightness_config is None else brightness_config
    if not isinstance(brightness_config, BrightnessConfig):
        raise ValueError("A BrightnessConfig is required")
    if band_config is not None and not isinstance(band_config, BandConfig):
        raise ValueError("A BandConfig is required")
    source = Path(source).resolve()
    lineage = None
    if region_receipt is None and expected_region_receipt_sha256 is not None:
        raise ValueError("Expected region hash requires a region receipt")
    if region_receipt is not None:
        receipt, child = verify_region(region_receipt, expected_receipt_sha256=expected_region_receipt_sha256)
        if child != source or receipt["modality"] != modality:
            raise ValueError("Region child source or modality disagrees with the measurement")
        lineage = {"receipt_path": str(Path(region_receipt).resolve()),
                   "receipt_sha256": expected_region_receipt_sha256,
                   **{k: receipt[k] for k in ("parent_source", "parent_file_sha256", "parent_pixel_sha256",
                                             "parent_shape_yx", "bbox_parent_level0_xyxy", "coordinate_mapping")},
                   "target_identity_status": receipt["target_identity_status"],
                   "region_identity_reviewed": False, "parent_crop_mapping_verified": True}
    methods = {
        "package_version": __version__,
        "axial_detector": {"engine_id": axial.ENGINE_ID, "settings": dict(axial.SETTINGS),
                           "confirm_native_rows": axial.CONFIRM_ROWS,
                           "overview_block_size": axial.F},
        "supporting_metrics": {"patch_size": 1024, "overview_max_dim": 1024,
                               "background_cells_per_axis": 8},
        "brightness_patterns": {"engine_id": BRIGHTNESS_ENGINE, "configuration": brightness_config.to_dict()},
    }
    if band_config is not None:
        methods["alternating_bands"] = {"engine_id": BANDS_ENGINE, "configuration": band_config.to_dict()}
    if lineage is not None:
        methods["source_region"] = {k: lineage[k] for k in ("receipt_sha256", "parent_file_sha256",
                                                            "parent_pixel_sha256", "bbox_parent_level0_xyxy", "target_identity_status")}
    methods_sha = hashlib.sha256(json.dumps(methods, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
    bands = measure_alternating_bands(native, config=band_config) if band_config is not None else None
    if fingerprint(source) != before or sha(source) != file_sha or native_hash(native) != pixels_sha:
        raise ValueError("Source file or decoded pixels changed during measurement")
    if lineage is not None and sha(lineage["parent_source"]) != lineage["parent_file_sha256"]:
        raise ValueError("Region parent changed during measurement")
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
        "source_region_lineage": lineage,
        "staging": {"file_sha256": file_sha, "source_access": "direct local read; no staging copy claimed"},
        "native_confirmed_patterns": len(unique), "scored_lines": None,
        "central_proxy_assessed": False, "central_proxy_is_reviewed_core_boundary": False,
        "lines": candidates, "metrics": metrics, "brightness_patterns": brightness, "alternating_bands": bands,
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
    parser.add_argument("--pattern-foreground-method", choices=["intensity_otsu", "hysteresis", "region_envelope"], default="intensity_otsu",
                        help="Explicit provisional brightness-region selection; neither method establishes core identity")
    parser.add_argument("--pattern-erosion-cells", type=int, help="Explicit erosion; default 3, or 0 for region_envelope")
    parser.add_argument("--region-receipt", type=Path, help="Verified parent-export mapping for a prepared analysis crop")
    parser.add_argument("--expected-region-receipt-sha256", help="Independently retained region receipt hash")
    parser.add_argument("--alternating-bands", action="store_true", help="Opt in to experimental bright/dark band geometry measurements")
    parser.add_argument("--band-region", help="Optional native X0,Y0,X1,Y1 rectangle; core identity remains unreviewed")
    parser.add_argument("--band-widths-px", help="Comma-separated native analysis widths; unresolved widths remain unassessed")
    parser.add_argument("--band-angles-deg", help="Comma-separated proposed line directions; native refinement remains bounded")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source.parent) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source image directory")
    config = BrightnessConfig(period_native_px=args.pattern_period_px, period_basis=args.pattern_period_basis,
                              foreground_method=args.pattern_foreground_method,
                              erosion_cells=(args.pattern_erosion_cells if args.pattern_erosion_cells is not None
                                             else 0 if args.pattern_foreground_method == "region_envelope" else 3))
    if not args.alternating_bands and any((args.band_region, args.band_widths_px, args.band_angles_deg)):
        parser.error("Band options require --alternating-bands")
    band_config = None
    if args.alternating_bands:
        options = {}
        if args.band_region:
            options["region_xyxy"] = tuple(int(v) for v in args.band_region.split(","))
        if args.band_widths_px:
            options["widths_native_px"] = tuple(int(v) for v in args.band_widths_px.split(","))
        if args.band_angles_deg:
            options["angles_degrees"] = tuple(float(v) for v in args.band_angles_deg.split(","))
        band_config = BandConfig(**options)
    result = measure_image(source, modality=args.modality, brightness_config=config, band_config=band_config,
                           region_receipt=args.region_receipt,
                           expected_region_receipt_sha256=args.expected_region_receipt_sha256)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "measurements.json", result)
    print(json.dumps({"record": str(output / "measurements.json"),
                      "modality": args.modality, "assessment": "measurements_only"}, indent=2))
