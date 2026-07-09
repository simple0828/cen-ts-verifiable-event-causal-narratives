from __future__ import annotations

from pathlib import Path

from cents.utils.io import read_json, write_json


def save_graph(graph: dict, path: str | Path) -> None:
    write_json(path, graph)


def load_graph(path: str | Path) -> dict:
    return read_json(path, default={"nodes": [], "edges": []})

