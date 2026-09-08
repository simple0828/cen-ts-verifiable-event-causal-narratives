from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_correlations(df: pd.DataFrame, max_lag: int = 3, threshold: float = 0.25) -> list[dict]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in ["text_rows"] if c in numeric], errors="ignore")
    edges: list[dict] = []
    cols = numeric.columns.tolist()
    for source in cols:
        for target in cols:
            if source == target:
                continue
            best = (0, 0.0)
            for lag in range(1, max_lag + 1):
                corr = numeric[source].shift(lag).corr(numeric[target])
                if pd.notna(corr) and abs(corr) > abs(best[1]):
                    best = (lag, float(corr))
            if abs(best[1]) >= threshold:
                edges.append({
                    "source": source,
                    "target": target,
                    "edge_type": "var_to_var",
                    "polarity": "positive" if best[1] > 0 else "negative",
                    "lag": best[0],
                    "confidence": min(1.0, abs(best[1])),
                    "evidence_score": abs(best[1]),
                })
    return edges

