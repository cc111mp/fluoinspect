"""Portable preflight and fixed job assignments; no inference or cluster scheduler.

Each node receives whole exports and maintains sessions on its local filesystem.
The plan is reproducible metadata. It does not grant a distributed execution lease.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tempfile

from ..io.persistence import DEFAULT_BUDGET, atomic_json

PLAN_SCHEMA = "af-qc.zoom-deployment-plan.v1"
PROFILE_SCHEMA = "af-qc.zoom-node-profile.v1"
REMOTE_FILESYSTEMS = {"cifs", "smb3", "nfs", "nfs4", "9p", "fuse.sshfs", "fuse.rclone"}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def normalized_job(value):
    keys = {"asset_id", "relative_path", "source_file_sha256", "size_bytes"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Each job needs asset ID, relative source path, file SHA-256 and byte size")
    asset = value["asset_id"]
    if not isinstance(asset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", asset):
        raise ValueError("Invalid asset ID")
    relative = value["relative_path"]
    if (not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or PurePosixPath(relative).is_absolute()):
        raise ValueError("Source path must be a contained relative POSIX path")
    checksum = value["source_file_sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
        raise ValueError("Expected lowercase file SHA-256 required")
    if type(value["size_bytes"]) is not int or value["size_bytes"] <= 0:
        raise ValueError("Positive source byte size required")
    return {key: value[key] for key in sorted(keys)}


def plan_jobs(jobs, node_count=2):
    if type(node_count) is not int or not 1 <= node_count <= 64:
        raise ValueError("Node count must be an integer from 1 to 64")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("A nonempty verified source manifest is required")
    normalized = sorted((normalized_job(job) for job in jobs), key=lambda j: j["asset_id"])
    if len({j["asset_id"] for j in normalized}) != len(normalized):
        raise ValueError("Duplicate asset ID")
    if len({j["relative_path"] for j in normalized}) != len(normalized):
        raise ValueError("Duplicate source path")
    identity = {"schema": PLAN_SCHEMA, "node_count": node_count,
                "assignment_method": "sha256(asset_id) modulo node_count", "jobs": normalized}
    nodes = [{"node_id": i, "jobs": [], "total_source_bytes": 0} for i in range(node_count)]
    for job in normalized:
        owner = int(hashlib.sha256(job["asset_id"].encode()).hexdigest(), 16) % node_count
        nodes[owner]["jobs"].append(job)
        nodes[owner]["total_source_bytes"] += job["size_bytes"]
    return {"schema": PLAN_SCHEMA, "plan_identity_sha256": json_hash(identity),
            "assignment_method": identity["assignment_method"], "node_count": node_count,
            "source_count": len(normalized), "total_source_bytes": sum(j["size_bytes"] for j in normalized),
            "nodes": nodes, "inference_executed": False, "distributed_leases_implemented": False,
            "human_QC_decision": ""}


def jobs_from_records(directory):
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError("No per-export records found")
    jobs = []
    for path in paths:
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            raise ValueError(f"Incomplete input record: {path.name}")
        stage = record["staging"]
        if not (stage.get("source_unchanged_before_after_copy") is True
                and stage.get("copy_readback_sha256_matches_source_stream") is True):
            raise ValueError(f"Unverified source-copy record: {path.name}")
        jobs.append({"asset_id": record["asset_id"], "relative_path": record["relative_path"],
                     "source_file_sha256": stage["file_sha256"], "size_bytes": stage["bytes_copied"]})
    return jobs


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("Unsupported deployment plan")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(n, dict) for n in nodes):
        raise ValueError("Invalid plan nodes")
    jobs = [job for node in nodes for job in node["jobs"]]
    expected = plan_jobs(jobs, plan["node_count"])
    if plan != expected:
        raise ValueError("Plan identity, assignments or counts disagree with its source manifest")
    return plan


def filesystem_type(path, mountinfo="/proc/self/mountinfo"):
    """Longest matching Linux mount; resolve symlinks before classification."""
    path = Path(path).resolve()
    candidates = []
    for line in Path(mountinfo).read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        root = Path(mount)
        if path == root or path.is_relative_to(root):
            candidates.append((len(root.parts), right.split()[0]))
    return max(candidates)[1] if candidates else None


def validate_profile(profile):
    required = {"schema", "node_id", "node_count", "paths", "execution", "review_budget", "inference"}
    if not isinstance(profile, dict) or set(profile) != required or profile["schema"] != PROFILE_SCHEMA:
        raise ValueError("Unsupported node profile or unexpected fields")
    count, node = profile["node_count"], profile["node_id"]
    if type(count) is not int or not 1 <= count <= 64 or type(node) is not int or not 0 <= node < count:
        raise ValueError("Node ID must be within the configured node count")
    paths = profile["paths"]
    if not isinstance(paths, dict) or set(paths) != {"local_cache_root", "local_work_root", "source_root", "publish_root"}:
        raise ValueError("Profile needs explicit cache, work, source and publication paths")
    for key, value in paths.items():
        if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
            raise ValueError(f"{key} must be an absolute path or null")
    execution = profile["execution"]
    if execution != {"active_sessions_per_node": 1, "cpu_threads_per_process": 1}:
        raise ValueError("Initial deployment requires one active session and one CPU thread per process")
    budget = profile["review_budget"]
    if (not isinstance(budget, dict) or set(budget) != set(DEFAULT_BUDGET)
            or any(type(v) is not int for v in budget.values())
            or not 2 <= budget["max_views"] <= 64
            or not 64 <= budget["max_display_dimension"] <= 2048
            or not 1 <= budget["max_raw_crop_pixels"] <= 16 * 1024 * 1024):
        raise ValueError("Invalid bounded review budget")
    inference = profile["inference"]
    inference_fields = {"endpoint", "model_id", "model_revision", "quantization", "prompt_sha256",
                        "image_preprocessing_revision", "request_timeout_seconds", "max_attempts"}
    if not isinstance(inference, dict) or set(inference) != inference_fields:
        raise ValueError("Inference settings require explicit model, prompt and preprocessing provenance")
    for key in inference_fields - {"request_timeout_seconds", "max_attempts"}:
        if inference[key] is not None and (not isinstance(inference[key], str) or not inference[key].strip()):
            raise ValueError(f"Invalid inference setting: {key}")
    if (type(inference["request_timeout_seconds"]) is not int or not 1 <= inference["request_timeout_seconds"] <= 600
            or type(inference["max_attempts"]) is not int or not 1 <= inference["max_attempts"] <= 3):
        raise ValueError("Inference timeouts and attempts must be bounded")
    if inference["prompt_sha256"] is not None and not re.fullmatch(r"[a-f0-9]{64}", inference["prompt_sha256"]):
        raise ValueError("Invalid prompt checksum")
    return profile


def preflight(profile):
    """Check CPU tools and local paths. No model endpoint or NAS is contacted."""
    profile = validate_profile(profile)
    blockers = []
    mounts = {}
    versions = {}
    for package in ("numpy", "Pillow", "tifffile", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package == "opencv-python-headless":
                try:
                    versions["opencv-python"] = importlib.metadata.version("opencv-python")
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
            blockers.append(f"Missing dependency: {package}")
    if platform.system() != "Linux" or sys.version_info < (3, 11):
        blockers.append("Linux and Python 3.11 or newer required")
    if platform.system() == "Linux":
        for key in ("local_cache_root", "local_work_root"):
            value = profile["paths"][key]
            if value is None:
                blockers.append(f"Configure {key}")
                continue
            try:
                fstype = filesystem_type(value)
                mounts[key] = fstype
                if fstype is None or fstype in REMOTE_FILESYSTEMS or fstype.startswith("fuse."):
                    blockers.append(f"{key} must use a verified local filesystem; detected {fstype}")
                    continue
                path = Path(value).resolve()
                if not path.is_dir():
                    blockers.append(f"Create {key}: {path}")
                    continue
                with tempfile.TemporaryFile(dir=path) as stream:
                    stream.write(b"af-qc-preflight")
                    stream.flush()
            except OSError as error:
                blockers.append(f"Cannot access {key}: {error}")
        cache, work = (profile["paths"][key] for key in ("local_cache_root", "local_work_root"))
        if cache and work:
            a, b = Path(cache).resolve(), Path(work).resolve()
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                blockers.append("Cache and work directories must be separate, non-nested paths")
    return {"schema": "af-qc.zoom-preflight.v1", "node_id": profile["node_id"],
            "architecture": platform.machine(), "python": platform.python_version(),
            "dependency_versions": versions, "local_filesystems": mounts,
            "cpu_tooling_ready": not blockers, "blockers": blockers,
            "inference_settings_complete": all(profile["inference"][key] for key in
                 ("endpoint", "model_id", "model_revision", "quantization", "prompt_sha256", "image_preprocessing_revision")),
            "inference_adapter_implemented": False, "inference_ready": False,
            "gpu_hardware_validated": False, "nas_contacted": False,
            "profile_sha256": json_hash(profile)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON list of verified source entries")
    source.add_argument("--records", type=Path, help="Completed per-export source-copy records")
    plan.add_argument("--nodes", type=int, default=2)
    plan.add_argument("--output", type=Path, required=True, help="Fresh plan directory")
    check = sub.add_parser("preflight")
    check.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        jobs = jobs_from_records(args.records) if args.records else json.loads(args.manifest.read_text())
        result = plan_jobs(jobs, args.nodes)
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "deployment_plan.json", result)
        for node in result["nodes"]:
            atomic_json(args.output / f"node-{node['node_id']:03d}.json",
                        {"plan_identity_sha256": result["plan_identity_sha256"], **node})
        print(json.dumps({key: value for key, value in result.items() if key != "nodes"}, indent=2))
    else:
        result = preflight(json.loads(args.profile.read_text()))
        print(json.dumps(result, indent=2))
        if not result["cpu_tooling_ready"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
