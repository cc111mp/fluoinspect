"""Provider-independent contract for future vision backends; no implementation."""
from typing import Any, Mapping, Protocol


class VisionBackend(Protocol):
    async def review(self, evidence: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
        """Return observations or requests; do not assign human acceptance."""
        ...
