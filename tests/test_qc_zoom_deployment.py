"""Job ownership, portable profile and local-filesystem readiness contracts."""
import copy
import json
from pathlib import Path

import pytest

from fluoinspect.deployment import plan as deploy
from fluoinspect.io.persistence import atomic_json

PROFILE = Path(__file__).resolve().parents[1] / "configs/node.example.json"


def jobs(count=80):
    return [{"asset_id": f"image-{i:04d}", "relative_path": f"series/AF {i}.tif",
             "source_file_sha256": f"{i:064x}", "size_bytes": i + 100} for i in range(count)]


def profile(tmp_path):
    value = json.loads(PROFILE.read_text())
    for key, directory in (("local_cache_root", "cache"), ("local_work_root", "work")):
        path = tmp_path / directory
        path.mkdir()
        value["paths"][key] = str(path)
    value["paths"]["source_root"] = "/unreachable-nas/source"
    value["paths"]["publish_root"] = "/unreachable-nas/results"
    return value


def owners(plan):
    return {j["asset_id"]: node["node_id"] for node in plan["nodes"] for j in node["jobs"]}


def test_plan_is_disjoint_complete_and_order_independent():
    inputs = jobs()
    first = deploy.plan_jobs(inputs)
    assert first == deploy.plan_jobs(list(reversed(inputs)))
    assigned = [j for node in first["nodes"] for j in node["jobs"]]
    assert sorted(assigned, key=lambda j: j["asset_id"]) == inputs
    assert len(owners(first)) == len(inputs)
    assert all(node["jobs"] for node in first["nodes"])
    assert first["total_source_bytes"] == sum(j["size_bytes"] for j in inputs)
    assert deploy.validate_plan(first) == first
    assert not first["inference_executed"] and first["human_QC_decision"] == ""


def test_existing_ownership_survives_added_inputs_and_hash_changes_reidentify_plan():
    first = deploy.plan_jobs(jobs(40))
    larger = deploy.plan_jobs(jobs(80))
    assert all(owners(larger)[asset] == owner for asset, owner in owners(first).items())
    changed = jobs(40)
    changed[0]["source_file_sha256"] = "f" * 64
    assert deploy.plan_jobs(changed)["plan_identity_sha256"] != first["plan_identity_sha256"]


@pytest.mark.parametrize("key", ["asset_id", "relative_path"])
def test_duplicate_inputs_are_refused(key):
    inputs = jobs(2)
    inputs[1][key] = inputs[0][key]
    with pytest.raises(ValueError, match="Duplicate"):
        deploy.plan_jobs(inputs)


@pytest.mark.parametrize("path", ["../outside.tif", "/absolute.tif", "folder/../outside.tif",
                                  "folder//image.tif", "folder\\image.tif", "./image.tif"])
def test_source_path_cannot_escape_or_be_ambiguous(path):
    inputs = jobs(1)
    inputs[0]["relative_path"] = path
    with pytest.raises(ValueError, match="contained relative"):
        deploy.plan_jobs(inputs)


@pytest.mark.parametrize("field,value", [("asset_id", "../bad"), ("source_file_sha256", "abc"),
                                         ("size_bytes", True), ("size_bytes", 0)])
def test_manifest_metadata_must_be_valid(field, value):
    inputs = jobs(1)
    inputs[0][field] = value
    with pytest.raises(ValueError):
        deploy.plan_jobs(inputs)


def test_tampered_assignment_is_rejected_even_with_unchanged_identity():
    plan = deploy.plan_jobs(jobs())
    plan["nodes"][1]["jobs"].append(plan["nodes"][0]["jobs"].pop())
    with pytest.raises(ValueError, match="assignments"):
        deploy.validate_plan(plan)


def test_from_records_requires_successful_verified_copy(tmp_path):
    record = {"status": "complete", "asset_id": "image-0000", "relative_path": "series/AF 0.tif",
              "staging": {"file_sha256": "0" * 64, "bytes_copied": 100,
                          "source_unchanged_before_after_copy": True,
                          "copy_readback_sha256_matches_source_stream": True}}
    path = tmp_path / "image-0000.json"
    path.write_text(json.dumps(record))
    assert deploy.jobs_from_records(tmp_path) == jobs(1)
    record["staging"]["copy_readback_sha256_matches_source_stream"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unverified"):
        deploy.jobs_from_records(tmp_path)
    record["status"] = "failed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incomplete"):
        deploy.jobs_from_records(tmp_path)


def test_preflight_only_checks_local_paths_and_keeps_inference_unready(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    contacted = []

    def mount_type(path):
        assert str(path) in {settings["paths"]["local_cache_root"], settings["paths"]["local_work_root"]}
        contacted.append(str(path))
        return "ext4"

    monkeypatch.setattr(deploy, "filesystem_type", mount_type)
    result = deploy.preflight(settings)
    assert result["cpu_tooling_ready"] and len(contacted) == 2
    assert not result["nas_contacted"] and not result["inference_ready"]
    assert not result["gpu_hardware_validated"]


def test_remote_working_storage_is_refused(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    monkeypatch.setattr(deploy, "filesystem_type", lambda _: "cifs")
    result = deploy.preflight(settings)
    assert not result["cpu_tooling_ready"]
    assert len(result["blockers"]) == 2 and all("local filesystem" in b for b in result["blockers"])


def test_nested_cache_and_work_is_refused(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    settings["paths"]["local_work_root"] = settings["paths"]["local_cache_root"]
    monkeypatch.setattr(deploy, "filesystem_type", lambda _: "ext4")
    result = deploy.preflight(settings)
    assert any("separate, non-nested" in b for b in result["blockers"])


def test_unconfigured_profile_is_an_explicit_blocker():
    settings = json.loads(PROFILE.read_text())
    result = deploy.preflight(settings)
    assert not result["cpu_tooling_ready"]
    assert {"Configure local_cache_root", "Configure local_work_root"}.issubset(result["blockers"])


def test_dependency_failure_is_reported_without_importing_image_library(tmp_path, monkeypatch):
    settings = profile(tmp_path)
    real_version = deploy.importlib.metadata.version

    def missing_numpy(package):
        if package == "numpy":
            raise deploy.importlib.metadata.PackageNotFoundError(package)
        return real_version(package)

    monkeypatch.setattr(deploy.importlib.metadata, "version", missing_numpy)
    monkeypatch.setattr(deploy, "filesystem_type", lambda _: "ext4")
    assert "Missing dependency: numpy" in deploy.preflight(settings)["blockers"]


def test_filesystem_classification_uses_resolved_longest_mount(tmp_path):
    local = tmp_path / "local"
    remote = local / "remote share"
    remote.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(remote, target_is_directory=True)
    mountinfo = tmp_path / "mountinfo"
    escaped = str(remote).replace(" ", "\\040")
    mountinfo.write_text(f"1 0 0:1 / / rw - ext4 /dev/root rw\n"
                        f"2 1 0:2 / {escaped} rw - cifs //nas/share rw\n")
    assert deploy.filesystem_type(local, mountinfo) == "ext4"
    assert deploy.filesystem_type(alias / "session", mountinfo) == "cifs"


@pytest.mark.parametrize("change", [{"node_id": 2}, {"node_count": True},
                                   {"review_budget": {"max_views": 999}}])
def test_invalid_profile_settings_are_refused(tmp_path, change):
    settings = profile(tmp_path)
    settings.update(change)
    with pytest.raises(ValueError):
        deploy.validate_profile(settings)


def test_json_failure_preserves_previous_receipt_and_cleans_temporary_file(tmp_path):
    target = tmp_path / "receipt.json"
    atomic_json(target, {"valid": True})
    with pytest.raises(ValueError):
        atomic_json(target, {"bad": float("nan")})
    assert json.loads(target.read_text()) == {"valid": True}
    assert list(tmp_path.iterdir()) == [target]


def test_assigned_creation_checks_node_and_staged_source_integrity(tmp_path):
    import numpy as np
    import tifffile
    from fluoinspect.investigation import session as zoom

    settings = profile(tmp_path)
    asset = "image-0000"
    staged = Path(settings["paths"]["local_cache_root"]) / f"{asset}.tif"
    tifffile.imwrite(staged, np.full((32, 40), 100, dtype=np.uint16), photometric="minisblack")
    manifest = [{"asset_id": asset, "relative_path": "series/original_AF.tif",
                 "source_file_sha256": zoom.sha(staged), "size_bytes": staged.stat().st_size}]
    plan = deploy.plan_jobs(manifest)
    owner = owners(plan)[asset]
    settings["node_id"] = 1 - owner
    output = Path(settings["paths"]["local_work_root"]) / "session"
    with pytest.raises(ValueError, match="not assigned"):
        zoom.create_assigned_session(plan, settings, asset, output)
    assert not output.exists()
    settings["node_id"] = owner
    zoom.create_assigned_session(plan, settings, asset, output)
    info = json.loads((output / "session.json").read_text())
    assert info["deployment"]["node_id"] == owner
    assert info["deployment"]["plan_identity_sha256"] == plan["plan_identity_sha256"]
    assert not info["deployment"]["inference_executed"] and info["human_QC_decision"] == ""
    with pytest.raises(ValueError, match="inside this node"):
        zoom.create_assigned_session(plan, settings, asset, tmp_path / "bad-output")
    changed = copy.deepcopy(plan)
    changed["nodes"][owner]["jobs"][0]["source_file_sha256"] = "f" * 64
    valid_changed = deploy.plan_jobs([j for n in changed["nodes"] for j in n["jobs"]])
    with pytest.raises(ValueError, match="Source hash differs"):
        zoom.create_assigned_session(valid_changed, settings, asset, output.parent / "changed-session")
