"""Bounded systematic detail planning and exact exported-area accounting.

Planned or exported views never establish model or human inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import session as zoom
from ..io.persistence import atomic_json
from ..io.native import MAX_NATIVE_PIXELS


def _validate_shape(shape_yx):
    if (not isinstance(shape_yx, (list, tuple)) or len(shape_yx) != 2
            or any(type(v) is not int or v <= 0 for v in shape_yx)
            or shape_yx[0] * shape_yx[1] > MAX_NATIVE_PIXELS):
        raise ValueError("Expected a bounded native YX shape")


def _intersect(first, second):
    box = [max(first[0], second[0]), max(first[1], second[1]),
           min(first[2], second[2]), min(first[3], second[3])]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _union_area(boxes):
    """Exact half-open rectangle union without a native-size raster allocation."""
    edges = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    area = 0
    for left, right in zip(edges, edges[1:]):
        spans = sorted((box[1], box[3]) for box in boxes if box[0] < right and box[2] > left)
        if not spans:
            continue
        start, end = spans[0]
        covered = 0
        for low, high in spans[1:]:
            if low > end:
                covered += end - start
                start = low
            end = max(end, high)
        area += (right - left) * (covered + end - start)
    return area


def _anchors(length, size, stride):
    last = max(0, length - size)
    regular = last // stride + 1
    return regular + int(last % stride != 0), last


def plan_tiles(shape_yx, *, target_box=None, tile_size=1024, stride=768, max_views=64):
    _validate_shape(shape_yx)
    if (type(tile_size) is not int or not 8 <= tile_size <= 2048
            or type(stride) is not int or not 1 <= stride <= tile_size
            or type(max_views) is not int or not 0 <= max_views <= 64):
        raise ValueError("Invalid tile size, stride or view budget")
    target = zoom.validate_box(target_box if target_box is not None
                               else [0, 0, shape_yx[1], shape_yx[0]], shape_yx)
    x0, y0, x1, y1 = target
    nx, last_x = _anchors(x1 - x0, tile_size, stride)
    ny, last_y = _anchors(y1 - y0, tile_size, stride)
    total = nx * ny
    count = min(max_views, total)
    indices = list(range(total)) if count == total else [i * total // count for i in range(count)]
    views = []
    for index in indices:
        iy, ix = divmod(index, nx)
        x = x0 + min(ix * stride, last_x)
        y = y0 + min(iy * stride, last_y)
        views.append({"request_id": f"systematic-{index:06d}", "kind": "detail",
                      "bbox_level0_xyxy": [x, y, min(x + tile_size, x1), min(y + tile_size, y1)],
                      "purpose": "Systematic detail coverage of an unreviewed analysis envelope",
                      "status": "planned", "model_inspection_logged": False})
    target_area = (x1 - x0) * (y1 - y0)
    area = _union_area([view["bbox_level0_xyxy"] for view in views])
    return {"schema": "fluoinspect.coverage-plan.v1", "source_shape_yx": list(shape_yx),
            "target_region_level0_xyxy": target, "region_identity_reviewed": False,
            "tile_size_native_px": tile_size, "stride_native_px": stride,
            "total_grid_regions": total, "planned_view_count": len(views),
            "omitted_grid_regions": total - len(views), "view_budget_truncated": count < total,
            "planned_native_area_px": area, "planned_native_area_fraction": area / target_area,
            "complete_grid_planned": count == total, "views": views,
            "model_inspection_established": False, "quality_decision": None}


def summarize_views(shape_yx, views, *, target_box=None):
    _validate_shape(shape_yx)
    if len(views) > 64:
        raise ValueError("View coverage exceeds the bounded session limit")
    target = zoom.validate_box(target_box if target_box is not None
                               else [0, 0, shape_yx[1], shape_yx[0]], shape_yx)
    preview_boxes, native_boxes = [], []
    for view in views:
        box = zoom.validate_box(view["bbox_level0_xyxy"], shape_yx)
        step = view["native_pixels_per_preview_pixel"]
        if type(step) is not int or step < 1:
            raise ValueError("Invalid recorded preview scale")
        clipped = _intersect(box, target)
        if clipped is not None:
            if step == 1:
                preview_boxes.append(clipped)
            if view.get("native_tiff") or view.get("raw_tiff"):
                native_boxes.append(clipped)
    total = (target[2] - target[0]) * (target[3] - target[1])
    native_area, preview_area = _union_area(native_boxes), _union_area(preview_boxes)
    return {"schema": "fluoinspect.coverage-summary.v1", "view_count": len(views),
            "target_region_level0_xyxy": target, "target_area_px": total,
            "region_identity_reviewed": False,
            "native_exported_area_px": native_area, "native_exported_area_fraction": native_area / total,
            "native_resolution_preview_area_px": preview_area,
            "native_resolution_preview_area_fraction": preview_area / total,
            "model_inspected_area_fraction": None,
            "full_resolution_image_inspection_established": False,
            "note": "Areas are geometric unions of recorded exports; the caller verifies artifacts. Export does not establish inspection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--bbox", nargs=4, type=int)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--max-views", type=int, default=64)
    parser.add_argument("--export-views", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.max_views <= 64:
        parser.error("View budget must be between 0 and 64")
    destination = args.output.resolve()
    with zoom.session_owner(args.session):
        session, info, native = zoom._load(args.session)
        zoom.validate_artifacts(session, info)
        if (destination.is_relative_to(session)
                or destination.is_relative_to(Path(info["source"]).resolve().parent)):
            raise ValueError("Coverage output must remain outside source and session directories")
        if args.export_views and (info.get("state") == "sealed" or (session / "assistant_review.json").exists()):
            raise ValueError("Sealed sessions cannot export additional coverage views")
        remaining = max(0, info["budget"]["max_views"] - len(info["views"]))
        if args.tile_size ** 2 > info["budget"]["max_raw_crop_pixels"]:
            raise ValueError("Detail tiles exceed the session native-crop budget")
        plan = plan_tiles(native.shape, target_box=args.bbox, tile_size=args.tile_size,
                          stride=args.stride, max_views=min(args.max_views, remaining))
        plan["source_file_sha256"] = info["source_file_sha256"]
        plan["source_pixel_sha256"] = info["decoded_pixel_sha256"]
        plan["session_remaining_view_budget"] = remaining
        basis = {k: plan[k] for k in ("source_shape_yx", "target_region_level0_xyxy", "tile_size_native_px",
                                     "stride_native_px", "source_file_sha256", "source_pixel_sha256")}
        basis["requests"] = [{k: view[k] for k in ("request_id", "kind", "bbox_level0_xyxy", "purpose")}
                             for view in plan["views"]]
        plan["plan_identity"] = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "coverage_plan.json", plan)
    if args.export_views:
        for view in plan["views"]:
            try:
                zoom.request_view(session, view["bbox_level0_xyxy"], kind="detail", purpose=view["purpose"],
                                  request_id=plan["plan_identity"][:16] + "-" + view["request_id"])
                view["status"] = "exported"
            except Exception as error:
                view["status"] = "export_failed"
                plan["export_error"] = str(error)
                atomic_json(destination / "coverage_plan.json", plan)
                raise
            atomic_json(destination / "coverage_plan.json", plan)
    with zoom.session_owner(session):
        session, info, _ = zoom._load(session)
        zoom.validate_artifacts(session, info)
        summary = summarize_views(info["shape_yx"], info["views"], target_box=args.bbox)
    atomic_json(destination / "exported_coverage.json", summary)
    print(json.dumps({"planned_views": plan["planned_view_count"],
                      "omitted_grid_regions": plan["omitted_grid_regions"],
                      "native_exported_area_fraction": summary["native_exported_area_fraction"],
                      "model_inspection_established": False}))
