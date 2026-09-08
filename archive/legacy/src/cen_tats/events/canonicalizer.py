from __future__ import annotations

from collections import Counter

from cen_tats.io_utils import stable_hash


def build_ontology(train_events: list[dict], min_support: int = 3) -> dict:
    counts = Counter(ev["event_type"] for ev in train_events)
    mapping = {event_type: (event_type if count >= min_support else "other") for event_type, count in counts.items()}
    mapping.setdefault("other", "other")
    return {
        "ontology_id": stable_hash({"counts": dict(counts), "min_support": min_support}),
        "min_support": min_support,
        "counts": dict(counts),
        "mapping": mapping,
        "built_from_split": "train_core",
    }


def apply_ontology(events: list[dict], ontology: dict) -> list[dict]:
    mapping = ontology.get("mapping", {})
    out: list[dict] = []
    for ev in events:
        item = dict(ev)
        item["raw_event_type"] = item["event_type"]
        item["event_type"] = mapping.get(item["event_type"], "other")
        out.append(item)
    return out
