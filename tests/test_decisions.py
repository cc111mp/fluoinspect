import asyncio
import copy
from dataclasses import replace
import json

import pytest

from fluoinspect.agents.decisions import (
    RESPONSE_SCHEMA, DecisionRequest, Question, ask_decisions, validate_response,
)
from fluoinspect.agents.triage import apply_triage_policy, make_triage_request


def request():
    return DecisionRequest.build({"source": "synthetic", "observations": ["possible band"]}, {
        "route": Question.choice("Choose next step", {"review": "Human review", "inspect": "More evidence"}),
        "priority": Question.score("Rank priority", {0: "Routine", 1: "Review", 2: "Priority"}),
        "concern": Question.binary("Is there evidence of a concern?"),
    })


def raw_response(req):
    return {"schema": RESPONSE_SCHEMA, "request_id": req.request_id, "model_id": "synthetic_fixture_no_inference",
            "answers": {"route": {"kind": "choice", "value": "review", "probabilities": {"review": 0.8, "inspect": 0.2}, "reported_confidence": 0.7},
                        "priority": {"kind": "score", "value": 1.3, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}},
                        "concern": {"kind": "binary", "value": 0.6}}}


def packet(active=True, integrity=True):
    return {"schema": "af-qc.evidence-packet.v1", "source": {"modality": "autofluorescence"},
            "verification": {"source_file_and_pixels_checked": integrity, "committed_view_hashes_checked": integrity},
            "views": [{"view_id": "view-001", "native_crop_exact_readback": True}],
            "next_crop_requests_allowed": active, "unassessed_checks": ["Correction references"],
            "coverage": {"full_resolution_image_inspection_established": False}}


def triage_response(req, choice="human_review", confidence=0.8):
    labels = dict(dict(req.questions)["next_step"].criteria)
    probabilities = {label: (1.0 if label == choice else 0.0) for label in labels}
    return validate_response(req, {"schema": RESPONSE_SCHEMA, "request_id": req.request_id,
                                  "model_id": "synthetic_fixture_no_inference", "answers": {
        "next_step": {"kind": "choice", "value": choice, "probabilities": probabilities, "reported_confidence": confidence},
        "concern_supported": {"kind": "binary", "value": 0.5},
        "review_priority": {"kind": "score", "value": 1, "probabilities": {"0": 0, "1": 1, "2": 0}},
    }})


def test_all_three_primitives_and_receipt_are_explicitly_uncalibrated():
    req = request()
    response = validate_response(req, raw_response(req))
    assert dict(response.answers)["priority"].value == pytest.approx(1.3)
    assert validate_response(req, response.to_dict()) == response
    assert response.receipt(req)["response_schema_validated"]
    assert not response.receipt(req)["confidence_calibrated_for_qc"]
    assert not response.receipt(req)["artifact_detection_accuracy_validated"]


def test_receipt_cannot_claim_validation_for_a_manually_modified_response():
    req = request()
    response = validate_response(req, raw_response(req))
    with pytest.raises(ValueError, match="identify the model"):
        replace(response, model_id="").receipt(req)


def test_request_is_immutable_and_binds_state_and_question_criteria():
    state = {"values": [1, 2]}
    options = {"a": "One", "b": "Two"}
    req = DecisionRequest.build(state, {"route": Question.choice("Pick", options)})
    original_id = req.request_id
    state["values"].append(3)
    options["a"] = "Changed"
    assert req.request_id == original_id
    assert json.loads(req.state_json)["values"] == [1, 2]
    changed = DecisionRequest.build(state, {"route": Question.choice("Pick", options)})
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_response(changed, {**raw_response(request()), "request_id": original_id})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01, 10**400])
def test_invalid_binary_probabilities_fail(value):
    req = request()
    raw = raw_response(req)
    raw["answers"]["concern"]["value"] = value
    with pytest.raises(ValueError):
        validate_response(req, raw)


@pytest.mark.parametrize("values", [{"review": 0.8}, {"review": 0.6, "inspect": 0.2},
                                    {"review": -0.1, "inspect": 1.1},
                                    {"review": 0.8, "inspect": 0.2, "accept": 0.0}])
def test_distribution_never_normalizes_or_invents_missing_options(values):
    req = request()
    raw = raw_response(req)
    raw["answers"]["route"]["probabilities"] = values
    with pytest.raises(ValueError):
        validate_response(req, raw)


@pytest.mark.parametrize("value", ["accept", "inspect", None])
def test_unknown_or_contradictory_choice_is_rejected(value):
    req = request()
    raw = raw_response(req)
    raw["answers"]["route"]["value"] = value
    with pytest.raises(ValueError):
        validate_response(req, raw)


def test_wrong_score_and_fabricated_acceptance_fields_are_rejected():
    req = request()
    raw = raw_response(req)
    raw["answers"]["priority"]["value"] = 2
    with pytest.raises(ValueError, match="Score"):
        validate_response(req, raw)
    raw = raw_response(req)
    raw["answers"]["route"]["human_QC_decision"] = "accepted"
    with pytest.raises(ValueError, match="typed fields"):
        validate_response(req, raw)


def test_missing_answers_and_malformed_question_contracts_fail():
    req = request()
    raw = raw_response(req)
    del raw["answers"]["concern"]
    with pytest.raises(ValueError, match="exactly match"):
        validate_response(req, raw)
    with pytest.raises(ValueError):
        Question.choice("Pick", {"only": "One"})
    with pytest.raises(ValueError):
        Question.score("Rank", {True: "One", 2: "Two"})
    with pytest.raises(ValueError):
        DecisionRequest.build({"score": float("nan")}, {"q": Question.binary("Test")})


def test_backend_invocation_validates_returned_values_without_a_default_provider():
    class FixtureBackend:
        async def decide(self, req):
            return raw_response(req)

    req = request()
    result = asyncio.run(ask_decisions(FixtureBackend(), req))
    assert result.model_id == "synthetic_fixture_no_inference"


def test_available_actions_follow_integrity_sealing_and_budget():
    active = make_triage_request(packet(), remaining_views=1,
                                  observations={"view-001": ["Synthetic possible intensity band"]})
    labels = dict(dict(active.questions)["next_step"].criteria)
    assert "gather_evidence" in labels and "record_concern" in labels
    for p, budget in [(packet(False), 1), (packet(), 0), (packet(integrity=False), 1)]:
        req = make_triage_request(p, remaining_views=budget)
        assert "gather_evidence" not in dict(dict(req.questions)["next_step"].criteria)
    assert not {"accept", "pass", "reject"} & set(labels)
    with pytest.raises(ValueError):
        make_triage_request(packet(), remaining_views=True)


def test_crop_existence_alone_does_not_establish_a_concern():
    req = make_triage_request(packet(), remaining_views=1)
    assert "record_concern" not in dict(dict(req.questions)["next_step"].criteria)
    observed = make_triage_request(packet(), observations={"view-001": ["Synthetic possible band"]})
    assert "record_concern" in dict(dict(observed.questions)["next_step"].criteria)
    assert observed.request_id != req.request_id


def test_observation_references_must_exist_in_the_packet():
    with pytest.raises(ValueError, match="known recorded view"):
        make_triage_request(packet(), observations={"invented-view": ["Concern"]})


def test_confident_decision_cannot_approve_quality_or_missing_integrity():
    req = make_triage_request(packet(integrity=False))
    result = apply_triage_policy(req, triage_response(req, confidence=1.0)).to_dict()
    assert result["action"] == "unassessed" and result["human_QC_decision"] == ""
    assert not result["automatic_quality_acceptance"]
    assert not result["confidence_calibrated_for_qc"]


def test_operational_cutoff_routes_low_confidence_to_review_and_preserves_unknown_checks():
    req = make_triage_request(packet(), remaining_views=1)
    result = apply_triage_policy(req, triage_response(req, "gather_evidence", 0.2),
                                 minimum_reported_confidence=0.5)
    assert result.action == "human_review"
    assert "unassessed_checks_preserved" in result.reasons
    changed_packet = copy.deepcopy(packet())
    changed_packet["source"]["modality"] = "fluorescence"
    different = make_triage_request(changed_packet)
    with pytest.raises(ValueError, match="identity mismatch"):
        apply_triage_policy(different, triage_response(req))


def test_custom_question_cannot_add_quality_acceptance_to_qc_policy():
    original = make_triage_request(packet())
    questions = dict(original.questions)
    questions["next_step"] = Question.choice("Pick", {"accept": "Approve image quality", "human_review": "Review"})
    custom = DecisionRequest.build(json.loads(original.state_json), questions)
    with pytest.raises(ValueError, match="not available"):
        apply_triage_policy(custom, triage_response(custom, "accept", 1.0))
