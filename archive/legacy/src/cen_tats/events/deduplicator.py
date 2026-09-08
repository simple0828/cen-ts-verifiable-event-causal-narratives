from __future__ import annotations

from collections import defaultdict

from cen_tats.io_utils import stable_hash


def deduplicate_events(events: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        key = (ev.get("time_index"), ev.get("actor"), ev.get("action"), ev.get("event_type"))
        groups[key].append(ev)
    output: list[dict] = []
    for key, items in groups.items():
        canonical_id = stable_hash({"dedup_key": key})
        for ev in items:
            item = dict(ev)
            item["canonical_event_id"] = canonical_id
            item["duplicate_count"] = len(items) - 1
            item["independent_source_count"] = len(set(x.get("source", "") for x in items))
            output.append(item)
    return output
