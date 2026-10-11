"""Code-owned review routing; no detector outcome assigns quality acceptance."""


def review_triage(*, periodic_state="unassessed", local_artifacts=None, target_identity_status="unreviewed"):
    if periodic_state not in {"flagged", "unflagged", "unassessed", "unverified"}:
        raise ValueError("Invalid periodic screen state")
    if target_identity_status not in {"unreviewed", "unresolved"}:
        raise ValueError("Invalid target identity status")
    reasons = []
    count = 0
    if local_artifacts is None:
        reasons.append("local_artifact_checks_not_run")
    else:
        if (local_artifacts.get("schema") != "fluoinspect.local-artifacts.v1"
                or local_artifacts.get("quality_decision") is not None
                or local_artifacts.get("artifact_accuracy_validated") is not False):
            raise ValueError("Invalid local artifact evidence")
        count = local_artifacts["candidate_count"]
        if count != len(local_artifacts["candidates"]):
            raise ValueError("Local candidate count disagrees")
        coverage = local_artifacts["coverage"]
        if coverage["tile_budget_truncated"] or coverage["detector_evaluated_area_fraction"] < 1:
            reasons.append("detector_coverage_incomplete")
        if coverage["candidate_proposals_omitted"]:
            reasons.append("candidate_budget_truncated")
        if count:
            reasons.append("local_artifact_candidates_require_review")
    if periodic_state == "flagged":
        reasons.append("periodic_candidate_requires_review")
    elif periodic_state in {"unassessed", "unverified"}:
        reasons.append("periodic_result_unassessed_or_uncalibrated")
    else:
        reasons.append("negative_periodic_screen_is_not_verified_clean")
    if target_identity_status == "unresolved":
        reasons.append("target_core_identity_unresolved")
    reasons.extend(["human_or_model_inspection_not_established", "artifact_accuracy_not_validated"])
    return {"schema": "fluoinspect.review-triage.v1", "verification_state": "unverified",
            "review_required": True, "route": "candidate_review" if count or periodic_state == "flagged"
            else "unflagged_region_review", "periodic_screen_state": periodic_state,
            "local_candidate_count": count, "target_identity_status": target_identity_status,
            "reasons": reasons, "quality_decision": None,
            "profile_decisions": {"quantitative_intensity": None, "morphology_modelling": None}}
