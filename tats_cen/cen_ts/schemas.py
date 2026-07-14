from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventRecord:
    timestamp: str
    source_text: str
    event_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    relation: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InverseEventRecord:
    timestamp: str
    event_text: str
    inverse_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationRecord:
    timestamp: str
    candidate_text: str
    is_supported: bool
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NarrativeRecord:
    timestamp: str
    narrative_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    description: str = ""


@dataclass(frozen=True)
class TextVariantRecord:
    timestamp: str
    source_text: str
    text: str
    mode: str
    text_column: str
    metadata: dict[str, Any] = field(default_factory=dict)
