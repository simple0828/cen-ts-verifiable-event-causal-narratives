from __future__ import annotations

from typing import Any

from .schemas import TextVariantRecord


class InverseEventGenerator:
    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        raise NotImplementedError("Inverse event generation is not implemented in P2.")
