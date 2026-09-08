from __future__ import annotations

from typing import Any

from .schemas import TextVariantRecord


class CausalGraphBuilder:
    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        raise NotImplementedError("Causal graph construction is not implemented in P2.")
