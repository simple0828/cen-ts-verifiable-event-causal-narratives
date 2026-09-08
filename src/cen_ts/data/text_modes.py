from __future__ import annotations

from enum import Enum
from typing import Iterable


class TextMode(str, Enum):
    RAW = "raw"
    EVENT = "event"


DEFAULT_TEXT_COLUMNS = {
    TextMode.RAW: "fact",
    TextMode.EVENT: "event_fact",
}


def parse_text_mode(value: str | TextMode) -> TextMode:
    if isinstance(value, TextMode):
        return value
    try:
        return TextMode(value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in TextMode)
        raise ValueError(f"Unsupported text_mode={value!r}; allowed: {allowed}") from exc


def default_text_column_for_mode(mode: str | TextMode) -> str:
    return DEFAULT_TEXT_COLUMNS[parse_text_mode(mode)]


def resolve_text_column(mode: str | TextMode, explicit_text_column: str | None = None) -> str:
    if explicit_text_column:
        return explicit_text_column
    return default_text_column_for_mode(mode)


def require_column(columns: Iterable[str], text_column: str) -> None:
    if text_column not in set(columns):
        raise ValueError(f"Required text column {text_column!r} is missing. No fallback to fact is allowed.")
