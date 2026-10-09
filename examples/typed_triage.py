"""Offline contract demonstration; the answers are synthetic, not model predictions."""
import json

from fluoinspect.agents.decisions import RESPONSE_SCHEMA, validate_response
from fluoinspect.agents.triage import apply_triage_policy, make_triage_request


def main():
    packet = {
        "schema": "af-qc.evidence-packet.v1", "source": {"modality": "autofluorescence"},
        "verification": {"source_file_and_pixels_checked": True, "committed_view_hashes_checked": True},
        "views": [{"view_id": "view-001", "native_crop_exact_readback": True}],
        "coverage": {"full_resolution_image_inspection_established": False},
        "next_crop_requests_allowed": True, "unassessed_checks": ["Correction references"],
    }
    request = make_triage_request(packet, remaining_views=2)
    criteria = dict(dict(request.questions)["next_step"].criteria)
    raw = {"schema": RESPONSE_SCHEMA, "request_id": request.request_id,
           "model_id": "synthetic_fixture_no_inference", "answers": {
        "next_step": {"kind": "choice", "value": "gather_evidence",
                      "probabilities": {name: 1.0 if name == "gather_evidence" else 0.0 for name in criteria},
                      "reported_confidence": 0.8},
        "concern_supported": {"kind": "binary", "value": 0.5},
        "review_priority": {"kind": "score", "value": 1.0, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
    }}
    response = validate_response(request, raw)
    recommendation = apply_triage_policy(request, response)
    print(json.dumps({"synthetic_example": True, "model_inference_executed": False,
                      "request": request.to_dict(), "validation_receipt": response.receipt(request),
                      "recommendation": recommendation.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
