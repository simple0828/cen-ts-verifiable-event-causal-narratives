from __future__ import annotations

from pathlib import Path

import pandas as pd

from cen_tats.data.aligned_dataset import AlignedDataset


def load_timemmd_domain(domain: str, processed_dir: str | Path = "data/processed/TimeMMD", target: str = "OT") -> AlignedDataset:
    path = Path(processed_dir) / f"{domain}.csv"
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} has no date column")
    if target not in df.columns:
        raise ValueError(f"{path} has no target column {target}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    fact = df["fact"].fillna("").astype(str) if "fact" in df else ""
    preds = df["preds"].fillna("").astype(str) if "preds" in df else ""
    existing = df["text"].fillna("").astype(str) if "text" in df else ""
    df["aux_text"] = (existing + " " + fact + " " + preds).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df.loc[df["aux_text"].eq("") | df["aux_text"].str.lower().eq("nan"), "aux_text"] = "No information available"
    df = df.sort_values("date").reset_index(drop=True)
    return AlignedDataset(name=domain, frame=df, target=target)


def dataset_audit(processed_dir: str | Path = "data/processed/TimeMMD", history: int = 24, horizon: int = 3) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(processed_dir).glob("*.csv")):
        if path.name.endswith("_stats.csv"):
            continue
        domain = path.stem
        ds = load_timemmd_domain(domain, processed_dir=processed_dir)
        df = ds.frame
        text = df[ds.text_col].fillna("").astype(str).str.strip()
        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in {"has_text", "text_rows"}]
        rows.append(
            {
                "dataset": domain,
                "timepoints": len(df),
                "start": str(df["date"].min().date()),
                "end": str(df["date"].max().date()),
                "numeric_variables": len(numeric_cols),
                "target": ds.target,
                "text_coverage": float((text != "").mean()),
                "effective_text_rows": int((text != "").sum()),
                "forecast_windows": max(0, len(df) - history - horizon + 1),
            }
        )
    return rows
