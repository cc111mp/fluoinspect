"""Code-owned QC routing over typed answers; never assigns image acceptance."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from .decisions import DecisionRequest, DecisionResponse, Question, _probability, validate_response

QC_ACTIONS = frozenset({"human_review", "unassessed", "gather_evidence", "record_concern"})


def make_triage_request(packet: Mapping[str, Any], *, remaining_views: int = 0,
                        observations: Mapping[str, list[str]] | None = None) -> DecisionRequest:
    if type(remaining_views) is not int or not 0 <= remaining_views <= 64:
        raise ValueError("Remaining view budget must be a bounded nonnegative integer")
    if not isinstance(packet, Mapping) or packet.get("schema") != "af-qc.evidence-packet.v1":
        raise ValueError("A supported evidence packet is required")
    verification = packet.get("verification", {})
    if not isinstance(verification, Mapping) or not isinstance(packet.get("views", []), list):
        raise ValueError("Packet verification and views must have the expected types")
    if any(not isinstance(v, Mapping) for v in packet.get("views", [])):
        raise ValueError("Each recorded view must be an object")
    observations = observations if observations is not None else {}
    if not isinstance(observations, Mapping) or len(observations) > 64:
        raise ValueError("Observations must be a bounded map of recorded view IDs to text lists")
    view_ids = {v.get("view_id") for v in packet.get("views", [])}
    for view_id, descriptions in observations.items():
        if (not isinstance(view_id, str) or view_id not in view_ids or not isinstance(descriptions, list)
                or not 1 <= len(descriptions) <= 16
                or any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in descriptions)):
            raise ValueError("Each observation must describe a known recorded view")
    measurements = packet.get("measurements")
    native_patterns = measurements.get("sharp_axial_patterns_anywhere", 0) if isinstance(measurements, Mapping) else 0
    if type(native_patterns) is not int or native_patterns < 0:
        raise ValueError("Native pattern count must be a nonnegative integer")
    integrity = (verification.get("source_file_and_pixels_checked") is True
                 and verification.get("committed_view_hashes_checked") is True)
    criteria = {"human_review": "Send observations and uncertainty for human review.",
                "unassessed": "Required evidence is missing or invalid; preserve unassessed status."}
    native_evidence = any(v.get("native_crop_exact_readback") is True for v in packet.get("views", []))
    concern_evidence = native_evidence and (bool(observations) or native_patterns > 0)
    if integrity and concern_evidence:
        criteria["record_concern"] = "Record a provisional visible concern; its cause and quality impact remain unvalidated."
    if integrity and packet.get("next_crop_requests_allowed") is True and remaining_views > 0:
        criteria["gather_evidence"] = "Request another bounded context, native-detail or comparison view."
    state = {
        "task": "fluoinspect_qc_triage_v1", "source": packet.get("source", {}),
        "verification": verification, "coverage": packet.get("coverage", {}),
        "views": [{k: v.get(k) for k in ("view_id", "kind", "bbox_level0_xyxy",
                  "native_pixels_per_preview_pixel", "native_crop_exact_readback", "model_inspection_logged")}
                  for v in packet.get("views", [])],
        "measurements": packet.get("measurements"), "tool_scopes": packet.get("tool_scopes", {}),
        "observations": dict(observations), "concern_evidence_supplied": concern_evidence,
        "unassessed_checks": packet.get("unassessed_checks", []),
        "integrity_established": integrity, "remaining_views": remaining_views,
    }
    return DecisionRequest.build(state, {
        "next_step": Question.choice("Choose one available next review step. No option approves image quality.", criteria),
        "concern_supported": Question.binary("Does the supplied evidence support a provisional concern? Missing inspection or references limit this judgment; do not infer cause from darkness alone."),
        "review_priority": Question.score("Rank human review priority from the supplied evidence; this is not a defect probability.",
                                           {0: "Routine review", 1: "Possible concern or missing evidence",
                                            2: "Potentially significant concern needing priority review"}),
    })


@dataclass(frozen=True)
class TriageRecommendation:
    action: str
    reasons: tuple[str, ...]
    request_id: str
    model_id: str
    reported_confidence: float | None

    def to_dict(self):
        return {"schema": "fluoinspect.triage-recommendation.v1", "action": self.action,
                "reasons": list(self.reasons), "request_id": self.request_id, "model_id": self.model_id,
                "reported_confidence": self.reported_confidence, "confidence_calibrated_for_qc": False,
                "human_QC_decision": "", "automatic_quality_acceptance": False}


def apply_triage_policy(request: DecisionRequest, response: DecisionResponse, *,
                        minimum_reported_confidence: float | None = None) -> TriageRecommendation:
    if not isinstance(response, DecisionResponse) or len(response.answers) != len({name for name, _ in response.answers}):
        raise ValueError("A complete typed decision response is required")
    response = validate_response(request, response.to_dict())
    cutoff = (_probability(minimum_reported_confidence) if minimum_reported_confidence is not None else None)
    if response.request_id != request.request_id:
        raise ValueError("Cannot route a response from another request")
    state = json.loads(request.state_json)
    if state.get("task") != "fluoinspect_qc_triage_v1":
        raise ValueError("QC policy requires a QC triage request")
    if {name for name, _ in response.answers} != {"next_step", "concern_supported", "review_priority"}:
        raise ValueError("QC policy requires all three triage answers")
    answer = dict(response.answers)["next_step"]
    allowed = dict(dict(request.questions)["next_step"].criteria)
    if answer.kind != "choice" or answer.value not in allowed or answer.value not in QC_ACTIONS:
        raise ValueError("Requested action is not available")
    action = str(answer.value)
    reasons = ["model_judgment_unvalidated_for_qc"]
    if state["integrity_established"] is not True:
        action = "unassessed"
        reasons.append("source_or_evidence_integrity_missing")
    elif action == "gather_evidence" and state["remaining_views"] <= 0:
        action = "human_review"
        reasons.append("view_budget_exhausted")
    elif action == "record_concern" and state.get("concern_evidence_supplied") is not True:
        action = "human_review"
        reasons.append("supporting_concern_evidence_missing")
    if action != "unassessed" and cutoff is not None:
        if answer.reported_confidence is None or answer.reported_confidence < cutoff:
            action = "human_review"
            reasons.append("reported_confidence_below_operational_cutoff")
    if state["unassessed_checks"]:
        reasons.append("unassessed_checks_preserved")
    return TriageRecommendation(action, tuple(reasons), request.request_id,
                                response.model_id, answer.reported_confidence)
