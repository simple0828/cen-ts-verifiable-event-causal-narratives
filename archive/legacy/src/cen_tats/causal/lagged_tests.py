from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def _lagged_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    rows = [x[max_lag - lag : len(x) - lag] for lag in range(1, max_lag + 1)]
    return np.column_stack(rows)


def distributed_lag_regression(event_values: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    if len(y) <= max_lag + 8 or np.nanstd(event_values) < 1e-12:
        return {"delta_r2": 0.0, "coef_sign": "neutral", "p_value": 1.0, "effect_size": 0.0}
    yy = y[max_lag:]
    y_lags = _lagged_matrix(y, max_lag)
    e_lags = _lagged_matrix(event_values, max_lag)
    restricted = LinearRegression().fit(y_lags, yy)
    unrestricted = LinearRegression().fit(np.column_stack([y_lags, e_lags]), yy)
    rss_r = float(np.sum((yy - restricted.predict(y_lags)) ** 2))
    rss_u = float(np.sum((yy - unrestricted.predict(np.column_stack([y_lags, e_lags]))) ** 2))
    tss = float(np.sum((yy - yy.mean()) ** 2)) + 1e-12
    delta_r2 = max(0.0, (rss_r - rss_u) / tss)
    q = max_lag
    df2 = max(1, len(yy) - 2 * max_lag - 1)
    f_stat = max(0.0, ((rss_r - rss_u) / max(q, 1)) / (rss_u / df2 + 1e-12))
    p_value = float(1.0 - stats.f.cdf(f_stat, q, df2))
    coef = unrestricted.coef_[-max_lag:]
    effect = float(np.nanmean(coef))
    return {
        "delta_r2": float(delta_r2),
        "coef_sign": "positive" if effect > 0 else "negative" if effect < 0 else "neutral",
        "p_value": p_value,
        "effect_size": effect,
        "f_stat": float(f_stat),
    }


def granger_style_test(event_values: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    return distributed_lag_regression(event_values, y, max_lag)


def event_study_bootstrap(event_values: np.ndarray, y: np.ndarray, lag_min: int, lag_max: int, n_boot: int = 200, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    event_idx = np.flatnonzero(np.abs(event_values) > 1e-12)
    diffs: list[float] = []
    for idx in event_idx:
        start = idx + lag_min
        end = min(len(y) - 1, idx + lag_max)
        if start < len(y) - 1 and end > start:
            diffs.append(float(np.nanmean(y[start : end + 1]) - y[idx]))
    if not diffs:
        return {"mean_change": 0.0, "ci_low": 0.0, "ci_high": 0.0, "direction_stability": 0.0, "support": 0}
    arr = np.asarray(diffs)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    mean_change = float(np.mean(arr))
    direction = 1 if mean_change >= 0 else -1
    stability = float(np.mean(np.sign(arr) == direction))
    return {
        "mean_change": mean_change,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "direction_stability": stability,
        "support": int(len(arr)),
    }


def permutation_test(event_values: np.ndarray, y: np.ndarray, max_lag: int, n_perm: int = 100, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    observed = distributed_lag_regression(event_values, y, max_lag)["delta_r2"]
    perm_scores = []
    for _ in range(n_perm):
        shuffled = rng.permutation(event_values)
        perm_scores.append(distributed_lag_regression(shuffled, y, max_lag)["delta_r2"])
    p = float((np.sum(np.asarray(perm_scores) >= observed) + 1) / (len(perm_scores) + 1))
    return {"observed_delta_r2": float(observed), "permutation_p_value": p, "n_perm": n_perm}
