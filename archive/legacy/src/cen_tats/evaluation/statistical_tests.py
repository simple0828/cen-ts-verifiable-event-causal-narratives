from __future__ import annotations

import numpy as np


def paired_bootstrap(a_errors: np.ndarray, b_errors: np.ndarray, n_boot: int = 1000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(a_errors, dtype=float).reshape(-1)
    b = np.asarray(b_errors, dtype=float).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diff = b[:n] - a[:n]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diff[idx])))
    return {"mean_diff": float(np.mean(diff)), "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)), "n_boot": n_boot}


def paired_permutation(a_errors: np.ndarray, b_errors: np.ndarray, n_perm: int = 1000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(a_errors, dtype=float).reshape(-1)
    b = np.asarray(b_errors, dtype=float).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return {"p_value": float("nan")}
    diff = b[:n] - a[:n]
    obs = abs(float(np.mean(diff)))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs(float(np.mean(diff * signs))) >= obs:
            count += 1
    return {"p_value": float((count + 1) / (n_perm + 1)), "n_perm": n_perm}
