"""Local native-crop tools for a vision-LLM-assisted AF review.

This module does not call an external model or assign human QC decisions.
All views retain source-coordinate provenance. Display transforms apply to PNGs;
native TIFF crops preserve uploaded-export uint16 values.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import tifffile

from ..io.native import read_native
from ..io.persistence import DEFAULT_BUDGET, atomic_json, fsync_directory

SCHEMA = "af-qc.zoom-session.v1"
ENGINE_REVISION = "zoom-tools-robust-v2"
FINDING_TYPES = {
    "parallel_bands", "seam_or_gap", "duplicate_structure", "background_shading",
    "clipping_concern", "blur_concern", "core_export_concern", "other",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    st = Path(path).stat()
    return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


@contextmanager
def session_owner(session):
    """Exclusive ownership on a worker's LOCAL filesystem; not a cluster lease."""
    session = Path(session)
    if not (session / "session.json").is_file():
        raise ValueError("Initialized local session required")
    with (session / ".session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller owns this local session") from None
        yield


def artifact_path(session, recorded):
    """Resolve portable artifact paths, including older local absolute receipts."""
    session = Path(session).resolve()
    path = Path(recorded)
    path = (path if path.is_absolute() else session / path).resolve()
    if not path.is_relative_to(session):
        raise ValueError("Evidence path escapes the session directory")
    return path


def native_hash(native):
    digest = hashlib.sha256()
    for y in range(0, native.shape[0], 512):
        digest.update(np.ascontiguousarray(native[y:y + 512], dtype="<u2").tobytes())
    return digest.hexdigest()


def validate_artifacts(session, info):
    for view in info["views"]:
        if sha(artifact_path(session, view["png"])) != view["png_sha256"]:
            raise ValueError("Evidence preview changed")
        if view["raw_tiff"] and sha(artifact_path(session, view["raw_tiff"])) != view["raw_tiff_sha256"]:
            raise ValueError("Evidence native crop changed")


def validate_box(box, shape_yx):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("BBox must contain four integer level-0 XY coordinates")
    x0, y0, x1, y1 = box
    height, width = shape_yx
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("BBox is empty or outside the source image")
    return list(box)


def block_mean(native, step):
    """Bounded block means with true partial-edge counts and no zero padding."""
    h, w = native.shape
    xs = np.arange(0, w, step)
    xcounts = np.minimum(step, w - xs)
    result = np.empty((math.ceil(h / step), len(xs)), dtype=np.float32)
    for index, y0 in enumerate(range(0, h, step)):
        stripe = native[y0:min(h, y0 + step)].astype(np.float64)
        summed = np.add.reduceat(stripe, xs, axis=1).sum(axis=0)
        result[index] = summed / (len(stripe) * xcounts)
    return result


def _load(session):
    session = Path(session)
    info = json.loads((session / "session.json").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError("Unsupported session schema")
    source = Path(info["source"])
    if fingerprint(source) != info["source_fingerprint"] or sha(source) != info["source_file_sha256"]:
        raise ValueError("Source changed since session creation")
    native = read_native(source)
    native.flags.writeable = False
    if fingerprint(source) != info["source_fingerprint"]:
        raise ValueError("Source changed during decode")
    if list(native.shape) != info["shape_yx"]:
        raise ValueError("Source geometry changed")
    if info.get("decoded_pixel_sha256") and native_hash(native) != info["decoded_pixel_sha256"]:
        raise ValueError("Decoded source pixels disagree with the session baseline")
    return session, info, native


def create_session(source, destination, *, expected_sha256=None, display_high=None,
                   label=None, budget=None, deployment=None, modality="autofluorescence"):
    if modality not in {"autofluorescence", "fluorescence"}:
        raise ValueError("Modality must be autofluorescence or fluorescence")
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Session output already exists; use a fresh path")
    if destination == source or source.is_relative_to(destination) or destination.is_relative_to(source.parent):
        raise ValueError("Session output must remain outside the source image directory")
    before = fingerprint(source)
    file_hash = sha(source)
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise ValueError("Source hash differs from the expected export")
    native = read_native(source)
    native.flags.writeable = False
    if fingerprint(source) != before:
        raise ValueError("Source changed during initialization")
    actual_budget = dict(DEFAULT_BUDGET)
    actual_budget.update(budget or {})
    if set(actual_budget) != set(DEFAULT_BUDGET) or any(type(v) is not int or v <= 0 for v in actual_budget.values()):
        raise ValueError("Invalid review budget")
    if not (2 <= actual_budget["max_views"] <= 64 and 64 <= actual_budget["max_display_dimension"] <= 2048
            and actual_budget["max_raw_crop_pixels"] <= 16 * 1024 * 1024):
        raise ValueError("Review budget exceeds bounded limits")
    if display_high is None:
        overview = block_mean(native, max(1, math.ceil(max(native.shape) / 1024)))
        display_high = max(float(np.percentile(overview, 99.5)), 1.0)
    if not isinstance(display_high, (int, float)) or not np.isfinite(display_high) or display_high <= 0:
        raise ValueError("Display high must be finite and positive")
    destination.mkdir(parents=True, mode=0o700)
    (destination / "views").mkdir(mode=0o700)
    info = {"schema": SCHEMA, "engine_revision": ENGINE_REVISION, "state": "active",
            "label": label or source.name, "source": str(source), "modality": modality,
            "source_file_sha256": file_hash, "source_fingerprint": before,
            "decoded_pixel_sha256": native_hash(native),
            "pixel_hash_byte_order": "little",
            "shape_yx": list(native.shape), "coordinate_system": "level-0 XY pixels; arrays YX",
            "display_window": {"low": 0, "high": float(display_high), "gamma_power": 0.5},
            "budget": actual_budget, "views": [], "human_QC_decision": "",
            "review_status": "unreviewed; assistant evidence is unvalidated",
            "physical_core_identity": None, "core_boundary_validated": False,
            "source_images_modified": False, "model_provider": "not configured"}
    if deployment is not None:
        info["deployment"] = deployment
    atomic_json(destination / "session.json", info)
    return request_view(destination, [0, 0, native.shape[1], native.shape[0]],
                        purpose="whole-export overview", kind="context")


def create_assigned_session(plan, profile, asset_id, destination, *, display_high=None, label=None, modality="autofluorescence"):
    """Create from this node's verified local cache; never stage or contact a model."""
    from ..deployment.plan import json_hash, preflight, validate_plan, validate_profile

    validate_plan(plan)
    validate_profile(profile)
    if profile["node_count"] != plan["node_count"]:
        raise ValueError("Node profile and deployment plan disagree on node count")
    node = plan["nodes"][profile["node_id"]]
    job = next((j for j in node["jobs"] if j["asset_id"] == asset_id), None)
    if job is None:
        raise ValueError("Asset is not assigned to this node")
    readiness = preflight(profile)
    if not readiness["cpu_tooling_ready"]:
        raise ValueError("Node preflight failed: " + "; ".join(readiness["blockers"]))
    work = Path(profile["paths"]["local_work_root"]).resolve()
    destination = Path(destination).resolve()
    if destination == work or not destination.is_relative_to(work):
        raise ValueError("Assigned session must be a fresh directory inside this node's local work root")
    cache = Path(profile["paths"]["local_cache_root"]).resolve()
    source = (cache / f"{asset_id}.tif").resolve()
    if not source.is_relative_to(cache):
        raise ValueError("Staged source escapes the local cache")
    if fingerprint(source)["size_bytes"] != job["size_bytes"]:
        raise ValueError("Staged source size differs from the deployment manifest")
    receipt = {"plan_identity_sha256": plan["plan_identity_sha256"], "node_id": profile["node_id"],
               "node_count": profile["node_count"], "asset_id": asset_id,
               "source_relative_path": job["relative_path"], "profile_sha256": json_hash(profile),
               "inference_settings": dict(profile["inference"]), "inference_executed": False}
    return create_session(source, destination, expected_sha256=job["source_file_sha256"],
                          display_high=display_high, label=label, budget=profile["review_budget"],
                          deployment=receipt, modality=modality)


def map_preview_box(view, preview_bbox):
    """Map a view's half-open preview rectangle to exact source block bounds."""
    box = validate_box(preview_bbox, view["preview_shape_yx"])
    x0, y0, x1, y1 = view["bbox_level0_xyxy"]
    a, b, c, d = box
    step = view["native_pixels_per_preview_pixel"]
    return [x0 + a * step, y0 + b * step, min(x1, x0 + c * step), min(y1, y0 + d * step)]


def request_view(session, bbox, *, purpose, kind="detail", parent_view_id=None,
                 coordinates="level0", request_id=None):
    """Serialize tool calls and make repeated identified requests idempotent."""
    with session_owner(session):
        return _request_view(session, bbox, purpose=purpose, kind=kind,
                             parent_view_id=parent_view_id, coordinates=coordinates,
                             request_id=request_id)


def _request_view(session, bbox, *, purpose, kind, parent_view_id, coordinates, request_id):
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 1000:
        raise ValueError("A concise investigation purpose is required")
    if kind not in {"context", "detail", "control"}:
        raise ValueError("View kind must be context, detail or control")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(type(v) is not int for v in bbox):
        raise ValueError("BBox must contain four integer coordinates")
    if coordinates not in {"level0", "preview"}:
        raise ValueError("Coordinates must be level0 or preview")
    if parent_view_id is not None and not isinstance(parent_view_id, str):
        raise ValueError("Parent view ID must be text or null")
    session, info, native = _load(session)
    if info.get("state") == "sealed" or (session / "assistant_review.json").exists():
        raise ValueError("Session is sealed; preserve evidence and create a new session")
    validate_artifacts(session, info)
    if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
                                   or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in request_id)):
        raise ValueError("Invalid request ID")
    request = {"bbox": list(bbox), "purpose": purpose, "kind": kind,
               "parent_view_id": parent_view_id, "coordinates": coordinates}
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    prior = next((v for v in info["views"] if request_id is not None and v.get("request_id") == request_id), None)
    if prior is not None:
        if prior.get("request_sha256") != request_hash:
            raise ValueError("Request ID reused with different arguments")
        return prior
    if len(info["views"]) >= info["budget"]["max_views"]:
        raise ValueError("Session view budget exhausted")
    if coordinates == "preview":
        parent = next((v for v in info["views"] if v["view_id"] == parent_view_id), None)
        if parent is None:
            raise ValueError("Known parent view required for preview coordinates")
        bbox = map_preview_box(parent, bbox)
    elif coordinates != "level0":
        raise ValueError("Coordinates must be level0 or preview")
    elif parent_view_id is not None and not any(v["view_id"] == parent_view_id for v in info["views"]):
        raise ValueError("Unknown parent view")
    bbox = validate_box(bbox, native.shape)
    x0, y0, x1, y1 = bbox
    crop = native[y0:y1, x0:x1]
    if kind in {"detail", "control"} and crop.size > info["budget"]["max_raw_crop_pixels"]:
        raise ValueError("Native detail crop too large; request context or a smaller bbox")
    view_id = f"view-{len(info['views']):03d}"
    maximum = info["budget"]["max_display_dimension"]
    step = max(1, math.ceil(max(crop.shape) / maximum))
    means = block_mean(crop, step)
    window = info["display_window"]
    pixels = (np.clip(means / window["high"], 0, 1) ** window["gamma_power"] * 255).astype(np.uint8)
    # An interrupted unregistered export is never overwritten or reused silently.
    # Only paths committed in session.json count as review evidence.
    artifact_id = view_id + "-" + uuid.uuid4().hex[:12]
    png = session / "views" / (artifact_id + ".png")
    Image.fromarray(pixels).save(png)
    raw = None
    if kind in {"detail", "control"}:
        raw = session / "views" / (artifact_id + ".tif")
        tifffile.imwrite(raw, crop, photometric="minisblack")
        restored = tifffile.imread(raw)
        if restored.dtype.kind != "u" or restored.dtype.itemsize != 2 or not np.array_equal(restored, crop):
            raise ValueError("Native crop readback changed source pixels")
    if fingerprint(info["source"]) != info["source_fingerprint"]:
        raise ValueError("Source changed during crop export")
    if info.get("decoded_pixel_sha256") and native_hash(native) != info["decoded_pixel_sha256"]:
        raise ValueError("Native pixels changed during crop creation")
    for path in (png, raw):
        if path is not None:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    fsync_directory(session / "views")
    view = {"view_id": view_id, "request_id": request_id, "request_sha256": request_hash,
            "kind": kind, "purpose": purpose,
            "parent_view_id": parent_view_id, "bbox_level0_xyxy": bbox,
            "preview_shape_yx": list(pixels.shape), "native_pixels_per_preview_pixel": step,
            "mapping": "Preview cells map to clipped native blocks beginning at bbox origin",
            "display_window": dict(window), "png": str(png.relative_to(session)), "png_sha256": sha(png),
            "raw_tiff": str(raw.relative_to(session)) if raw else None, "raw_tiff_sha256": sha(raw) if raw else None,
            "all_raw_crop_pixels_equal_source": True if raw else None,
            "pixel_count_raw_rectangle": int(crop.size), "source_pixel_values_modified": False}
    info["views"].append(view)
    atomic_json(session / "session.json", info)
    return view


def validate_report(session, report):
    """Validate localization/evidence references, not biological correctness."""
    with session_owner(session):
        return _validate_report(session, report)


def _validate_report(session, report):
    session, info, _ = _load(session)
    expected = {"schema", "observation_status", "findings", "unassessed_checks", "limitations"}
    if not isinstance(report, dict) or set(report) != expected or report["schema"] != "af-qc.assistant-review.v1":
        raise ValueError("Report requires the defined schema and fields")
    if report["observation_status"] not in {"concerns_in_views", "no_findings_in_views", "uncertain", "incomplete"}:
        raise ValueError("Observation status cannot assign image acceptance")
    if not isinstance(report["findings"], list) or len(report["findings"]) > 64:
        raise ValueError("Invalid findings list")
    for key in ("unassessed_checks", "limitations"):
        if (not isinstance(report[key], list) or len(report[key]) > 64
                or any(not isinstance(v, str) or len(v) > 2000 for v in report[key])):
            raise ValueError("Limitations and unassessed checks must be text lists")
    views = {v["view_id"]: v for v in info["views"]}
    for finding in report["findings"]:
        fields = {"type", "bbox_level0_xyxy", "evidence_view_ids", "observation", "cause_hypothesis", "certainty"}
        if not isinstance(finding, dict) or set(finding) != fields or finding["type"] not in FINDING_TYPES:
            raise ValueError("Invalid finding type/fields")
        if finding["certainty"] not in {"visible_pattern", "suspected", "uncertain"}:
            raise ValueError("Finding certainty describes observations, not accuracy probabilities")
        if not isinstance(finding["observation"], str) or not finding["observation"].strip():
            raise ValueError("Finding needs an observed description")
        if finding["cause_hypothesis"] is not None and not isinstance(finding["cause_hypothesis"], str):
            raise ValueError("Cause hypothesis must be text or null")
        box = validate_box(finding["bbox_level0_xyxy"], info["shape_yx"])
        evidence = finding["evidence_view_ids"]
        if (not isinstance(evidence, list) or not evidence or len(evidence) > 64
                or any(not isinstance(v, str) or v not in views for v in evidence)):
            raise ValueError("Finding requires known recorded evidence views")
        if not any(_contains(views[v]["bbox_level0_xyxy"], box) for v in evidence):
            raise ValueError("Finding bbox lies outside its evidence views")
    if report["findings"] and report["observation_status"] == "no_findings_in_views":
        raise ValueError("Report contradicts its findings")
    validate_artifacts(session, info)
    sealed = {**report, "source_file_sha256": info["source_file_sha256"],
              "shape_yx": info["shape_yx"], "recorded_view_ids": list(views),
              "human_QC_decision": "", "artifact_accuracy_validated": False,
              "report_geometry_and_evidence_validated": True,
              "review_status": "assistant observations; human review pending"}
    target = session / "assistant_review.json"
    if target.exists():
        raise FileExistsError("Preserve the existing report; use a fresh session for revisions")
    atomic_json(target, sealed)
    info["state"] = "sealed"
    info["assistant_review_sha256"] = sha(target)
    atomic_json(session / "session.json", info)
    return sealed


def inspect_session(session):
    """Audit committed local evidence without loading or touching the source TIFF."""
    with session_owner(session):
        session = Path(session).resolve()
        info = json.loads((session / "session.json").read_text())
        if info.get("schema") != SCHEMA:
            raise ValueError("Unsupported session schema")
        validate_artifacts(session, info)
        recorded = {artifact_path(session, v[k]) for v in info["views"]
                    for k in ("png", "raw_tiff") if v[k]}
        orphans = sorted(str(p.relative_to(session)) for p in (session / "views").iterdir()
                         if p.suffix in {".png", ".tif"} and p.resolve() not in recorded)
        target = session / "assistant_review.json"
        if target.exists():
            checksum = sha(target)
            if info.get("assistant_review_sha256") and checksum != info["assistant_review_sha256"]:
                raise ValueError("Sealed assistant report changed")
            state = "sealed" if info.get("state") == "sealed" else "report_present_metadata_pending"
        else:
            if info.get("state") == "sealed" or info.get("assistant_review_sha256"):
                raise ValueError("Sealed assistant report missing")
            state = "active"
        return {"state": state, "committed_views": len(info["views"]),
                "unregistered_files": orphans, "evidence_file_hashes_validated": True,
                "source_rechecked": False, "human_QC_decision": info["human_QC_decision"],
                "resume_allowed": state == "active",
                "note": "Unregistered files are excluded from evidence. Preserve interrupted reports; do not overwrite."}


def _contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--expected-sha256")
    create.add_argument("--display-high", type=float)
    create.add_argument("--label")
    create.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    assigned = sub.add_parser("create-assigned")
    assigned.add_argument("--plan", required=True, type=Path)
    assigned.add_argument("--profile", required=True, type=Path)
    assigned.add_argument("--asset-id", required=True)
    assigned.add_argument("--output", required=True, type=Path)
    assigned.add_argument("--display-high", type=float)
    assigned.add_argument("--label")
    assigned.add_argument("--modality", choices=["autofluorescence", "fluorescence"], default="autofluorescence")
    view = sub.add_parser("view")
    view.add_argument("--session", required=True, type=Path)
    view.add_argument("--bbox", required=True, type=int, nargs=4)
    view.add_argument("--purpose", required=True)
    view.add_argument("--kind", choices=["context", "detail", "control"], default="detail")
    view.add_argument("--coordinates", choices=["level0", "preview"], default="level0")
    view.add_argument("--parent-view-id")
    view.add_argument("--request-id")
    report = sub.add_parser("validate-report")
    report.add_argument("--session", required=True, type=Path)
    report.add_argument("--report", required=True, type=Path)
    status = sub.add_parser("status")
    status.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create_session(args.source, args.output, expected_sha256=args.expected_sha256,
                                display_high=args.display_high, label=args.label, modality=args.modality)
    elif args.command == "create-assigned":
        result = create_assigned_session(json.loads(args.plan.read_text()), json.loads(args.profile.read_text()),
                                          args.asset_id, args.output, display_high=args.display_high, label=args.label, modality=args.modality)
    elif args.command == "view":
        result = request_view(args.session, args.bbox, purpose=args.purpose, kind=args.kind,
                              parent_view_id=args.parent_view_id, coordinates=args.coordinates,
                              request_id=args.request_id)
    elif args.command == "validate-report":
        result = validate_report(args.session, json.loads(args.report.read_text()))
    else:
        result = inspect_session(args.session)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
