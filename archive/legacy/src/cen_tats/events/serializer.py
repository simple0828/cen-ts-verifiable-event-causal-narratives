from __future__ import annotations

from cen_tats.io_utils import read_jsonl, write_jsonl


def save_events(path: str, events: list[dict]) -> None:
    write_jsonl(path, events)


def load_events(path: str) -> list[dict]:
    return read_jsonl(path)
