from __future__ import annotations

from typing import Any

from .schemas import TextVariantRecord


class BidirectionalVerifier:
    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        raise NotImplementedError("Bidirectional verification is not implemented in P2.")
