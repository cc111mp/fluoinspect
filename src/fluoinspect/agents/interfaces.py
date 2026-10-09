"""Provider-independent contract for future vision backends; no implementation."""
from typing import Any, Mapping, Protocol

from .decisions import DecisionRequest


class VisionBackend(Protocol):
    async def review(self, evidence: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
        """Return observations or requests; do not assign human acceptance."""
        ...


class DecisionBackend(Protocol):
    async def decide(self, request: DecisionRequest) -> Mapping[str, Any]:
        """Return the typed response envelope; validation and routing stay in code."""
        ...
