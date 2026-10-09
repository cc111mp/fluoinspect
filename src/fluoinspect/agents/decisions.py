"""Typed decision contracts; schema correctness does not establish QC accuracy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

REQUEST_SCHEMA = "fluoinspect.decision-request.v1"
RESPONSE_SCHEMA = "fluoinspect.decision-response.v1"


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", value) is not None


def _number(value):
    if type(value) not in {int, float}:
        raise ValueError("A finite number is required; booleans are not numbers")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("A finite number is required") from None
    if not math.isfinite(number):
        raise ValueError("A finite number is required")
    return number


def _probability(value):
    value = _number(value)
    if not 0 <= value <= 1:
        raise ValueError("Probability must lie between zero and one")
    return value


@dataclass(frozen=True)
class Question:
    kind: str
    instructions: str
    criteria: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.kind not in {"choice", "score", "binary"}:
            raise ValueError("Question kind must be choice, score or binary")
        if not isinstance(self.instructions, str) or not self.instructions.strip() or len(self.instructions) > 2000:
            raise ValueError("A bounded, nonempty question instruction is required")
        if (type(self.criteria) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                or not all(isinstance(v, str) for v in pair) for pair in self.criteria)):
            raise ValueError("Criteria must be immutable string pairs")
        if self.kind == "binary":
            if self.criteria:
                raise ValueError("Binary questions do not define choice criteria")
            return
        if not 2 <= len(self.criteria) <= 255 or len({k for k, _ in self.criteria}) != len(self.criteria):
            raise ValueError("Two to 255 unique criteria are required")
        if any(not v.strip() or len(v) > 2000 for _, v in self.criteria):
            raise ValueError("Each criterion needs a bounded description")
        if self.kind == "choice" and any(not _identifier(k) for k, _ in self.criteria):
            raise ValueError("Choice identifiers must be bounded names")
        if self.kind == "score":
            if any(re.fullmatch(r"-?(0|[1-9][0-9]*)", k) is None for k, _ in self.criteria):
                raise ValueError("Score tiers must be integer labels")
            levels = [int(k) for k, _ in self.criteria]
            if levels != sorted(set(levels)) or any(abs(n) > 1000000 for n in levels):
                raise ValueError("Score tiers must be bounded and strictly increasing")

    @classmethod
    def choice(cls, instructions: str, criteria: Mapping[str, str]):
        return cls("choice", instructions, tuple(sorted(criteria.items())))

    @classmethod
    def score(cls, instructions: str, criteria: Mapping[int, str]):
        if any(type(key) is not int for key in criteria):
            raise ValueError("Score tiers must be integers")
        return cls("score", instructions, tuple((str(k), v) for k, v in sorted(criteria.items())))

    @classmethod
    def binary(cls, instructions: str):
        return cls("binary", instructions)

    def to_dict(self):
        return {"kind": self.kind, "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class DecisionRequest:
    state_json: str
    questions: tuple[tuple[str, Question], ...]

    def __post_init__(self):
        if not isinstance(self.state_json, str) or len(self.state_json.encode()) > 1024 * 1024:
            raise ValueError("State must be bounded JSON")
        state = json.loads(self.state_json)
        if not isinstance(state, dict):
            raise ValueError("Decision state must be a JSON object")
        # Reject NaN/Infinity even if constructed through Python's permissive decoder.
        json.dumps(state, allow_nan=False)
        if type(self.questions) is not tuple or not 1 <= len(self.questions) <= 64:
            raise ValueError("One to 64 immutable named questions are required")
        for pair in self.questions:
            if type(pair) is not tuple or len(pair) != 2 or not _identifier(pair[0]) or not isinstance(pair[1], Question):
                raise ValueError("Invalid named question")
        if len({name for name, _ in self.questions}) != len(self.questions):
            raise ValueError("Question names must be unique")

    @classmethod
    def build(cls, state: Mapping[str, Any], questions: Mapping[str, Question]):
        state_json = json.dumps(dict(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(state_json, tuple(sorted(questions.items())))

    @property
    def request_id(self):
        payload = {"schema": REQUEST_SCHEMA, "state": json.loads(self.state_json),
                   "questions": {name: question.to_dict() for name, question in self.questions}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()

    def to_dict(self):
        return {"schema": REQUEST_SCHEMA, "request_id": self.request_id,
                "state": json.loads(self.state_json),
                "questions": {name: question.to_dict() for name, question in self.questions}}


@dataclass(frozen=True)
class Answer:
    kind: str
    value: str | float
    probabilities: tuple[tuple[str, float], ...]
    reported_confidence: float | None

    def to_dict(self):
        value = {"kind": self.kind, "value": self.value, "reported_confidence": self.reported_confidence}
        if self.kind != "binary":
            value["probabilities"] = dict(self.probabilities)
        return value


@dataclass(frozen=True)
class DecisionResponse:
    request_id: str
    model_id: str
    answers: tuple[tuple[str, Answer], ...]

    def to_dict(self):
        if len(self.answers) != len({name for name, _ in self.answers}):
            raise ValueError("Answer names must be unique")
        return {"schema": RESPONSE_SCHEMA, "request_id": self.request_id, "model_id": self.model_id,
                "answers": {name: answer.to_dict() for name, answer in self.answers}}

    def receipt(self, request: DecisionRequest):
        validated = validate_response(request, self.to_dict())
        return {"schema": "fluoinspect.decision-receipt.v1", "response": validated.to_dict(),
                "response_schema_validated": True, "confidence_calibrated_for_qc": False,
                "artifact_detection_accuracy_validated": False}


def _distribution(raw, criteria):
    if not isinstance(raw, dict) or set(raw) != {name for name, _ in criteria}:
        raise ValueError("Probability labels must exactly match the declared criteria")
    result = tuple((name, _probability(raw[name])) for name, _ in criteria)
    if not math.isclose(math.fsum(p for _, p in result), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one; no silent normalization")
    return result


def validate_response(request: DecisionRequest, raw: Mapping[str, Any]) -> DecisionResponse:
    if not isinstance(raw, dict) or set(raw) != {"schema", "request_id", "model_id", "answers"}:
        raise ValueError("Response envelope requires the defined fields")
    if raw["schema"] != RESPONSE_SCHEMA or raw["request_id"] != request.request_id:
        raise ValueError("Response schema or request identity mismatch")
    if not isinstance(raw["model_id"], str) or not raw["model_id"].strip() or len(raw["model_id"]) > 256:
        raise ValueError("Response must identify the model")
    if not isinstance(raw["answers"], dict) or set(raw["answers"]) != {name for name, _ in request.questions}:
        raise ValueError("Answers must exactly match the requested questions")
    answers = []
    for name, question in request.questions:
        value = raw["answers"][name]
        required = {"kind", "value"} | ({"probabilities"} if question.kind != "binary" else set())
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"reported_confidence"}:
            raise ValueError("Answer requires the defined typed fields")
        if value["kind"] != question.kind:
            raise ValueError("Answer kind disagrees with question")
        confidence = value.get("reported_confidence")
        confidence = _probability(confidence) if confidence is not None else None
        probabilities = () if question.kind == "binary" else _distribution(value["probabilities"], question.criteria)
        if question.kind == "choice":
            selection = value["value"]
            if not isinstance(selection, str) or selection not in dict(question.criteria):
                raise ValueError("Choice is outside the declared options")
            if dict(probabilities)[selection] + 1e-6 < max(p for _, p in probabilities):
                raise ValueError("Choice contradicts the supplied probability distribution")
        elif question.kind == "score":
            selection = _number(value["value"])
            expected = math.fsum(int(label) * p for label, p in probabilities)
            if not math.isclose(selection, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Score must match the expected declared rubric tier")
        else:
            selection = _probability(value["value"])
        answers.append((name, Answer(question.kind, selection, probabilities, confidence)))
    return DecisionResponse(request.request_id, raw["model_id"], tuple(answers))


async def ask_decisions(backend, request: DecisionRequest) -> DecisionResponse:
    """Invoke a caller-supplied backend and validate its output; no default provider."""
    return validate_response(request, await backend.decide(request))
