from __future__ import annotations

from cen_tats.data.split_manager import V5Split


def assert_temporal_order(split: V5Split) -> None:
    ranges = [split.train_core, split.prompt_dev, split.model_val, split.test]
    for left, right in zip(ranges, ranges[1:]):
        if len(left) and len(right) and int(left[-1]) >= int(right[0]):
            raise AssertionError("Temporal split order is violated")


def assert_window_has_no_future_text(anchor: int, history_indices: list[int]) -> None:
    if history_indices and max(history_indices) > anchor:
        raise AssertionError("History text includes future timestamps")


def assert_inverse_uses_history_only(anchor: int, used_indices: list[int]) -> None:
    if used_indices and max(used_indices) > anchor:
        raise AssertionError("Inverse event input includes future series")


def assert_train_only_graph(edge_payload: dict) -> None:
    if edge_payload.get("split_used") != "train_core":
        raise AssertionError("Causal graph was not built only from train_core")
