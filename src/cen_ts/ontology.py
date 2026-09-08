from __future__ import annotations

from typing import Any


class EventOntology:
    def normalize(self, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Event ontology normalization is not implemented in P2.")
