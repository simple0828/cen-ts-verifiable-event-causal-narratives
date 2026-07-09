from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def _lag_matrix(values: np.ndarray, lags: int) -> np.ndarray:
    rows = []
    for lag in range(1, lags + 1):
        rows.append(values[lags - lag: len(values) - lag])
    return np.column_stack(rows)


def granger_lite_pair(x: np.ndarray, y: np.ndarray, max_lag: int = 3) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) <= max_lag + 8 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return {"f_stat": 0.0, "p_value": 1.0, "best_lag": max_lag, "effect": 0.0}
    y_target = y[max_lag:]
    y_lags = _lag_matrix(y, max_lag)
    x_lags = _lag_matrix(x, max_lag)
    restricted = LinearRegression().fit(y_lags, y_target)
    unrestricted = LinearRegression().fit(np.column_stack([y_lags, x_lags]), y_target)
    rss_r = float(np.sum((y_target - restricted.predict(y_lags)) ** 2))
    rss_u = float(np.sum((y_target - unrestricted.predict(np.column_stack([y_lags, x_lags]))) ** 2))
    q = max_lag
    df2 = max(1, len(y_target) - 2 * max_lag - 1)
    f_stat = max(0.0, ((rss_r - rss_u) / q) / max(rss_u / df2, 1e-12))
    p_value = float(stats.f.sf(f_stat, q, df2))
    corrs = [pd.Series(x).shift(lag).corr(pd.Series(y)) for lag in range(1, max_lag + 1)]
    best_idx = int(np.nanargmax(np.abs(np.nan_to_num(corrs)))) if corrs else 0
    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "best_lag": best_idx + 1,
        "effect": float(np.nan_to_num(corrs[best_idx] if corrs else 0.0)),
    }


def granger_lite_edges(df: pd.DataFrame, target: str = "OT", max_lag: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in ["text_rows", "has_text"] if c in numeric], errors="ignore")
    rows = []
    cols = numeric.columns.tolist()
    for source in cols:
        for dest in cols:
            if source == dest:
                continue
            score = granger_lite_pair(numeric[source].to_numpy(), numeric[dest].to_numpy(), max_lag=max_lag)
            rows.append({
                "source": source,
                "target": dest,
                "best_lag": score["best_lag"],
                "f_stat": score["f_stat"],
                "p_value": score["p_value"],
                "effect": score["effect"],
                "significant": bool(score["p_value"] < alpha),
                "target_is_ot": bool(dest == target),
            })
    columns = ["source", "target", "best_lag", "f_stat", "p_value", "effect", "significant", "target_is_ot"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["significant", "target_is_ot", "f_stat"], ascending=[False, False, False])
