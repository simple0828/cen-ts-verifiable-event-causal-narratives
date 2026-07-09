from __future__ import annotations


def performance_drop(original: float, perturbed: float) -> float:
    if abs(original) < 1e-12:
        return 0.0
    return float((perturbed - original) / abs(original))

