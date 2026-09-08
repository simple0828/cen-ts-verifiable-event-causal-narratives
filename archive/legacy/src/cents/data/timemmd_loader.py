from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cents.data.preprocessing import clean_text, infer_frequency, recommended_windows
from cents.utils.io import ensure_dir, write_json


def _csvs(path: Path) -> list[Path]:
    return sorted(path.glob("*.csv"))


def discover_domains(raw_dir: str | Path) -> list[str]:
    raw_dir = Path(raw_dir)
    numerical = raw_dir / "numerical"
    textual = raw_dir / "textual"
    if not numerical.exists() or not textual.exists():
        return []
    n_domains = {p.name for p in numerical.iterdir() if p.is_dir()}
    t_domains = {p.name for p in textual.iterdir() if p.is_dir()}
    return sorted(n_domains & t_domains)


def load_raw_domain(raw_dir: str | Path, domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    num_files = _csvs(raw_dir / "numerical" / domain)
    text_files = _csvs(raw_dir / "textual" / domain)
    if not num_files:
        raise FileNotFoundError(f"No numerical csv found for {domain}")
    numerical = pd.read_csv(num_files[0])
    text_parts = []
    for file in text_files:
        part = pd.read_csv(file)
        part["text_source"] = file.stem
        text_parts.append(part)
    textual = pd.concat(text_parts, ignore_index=True) if text_parts else pd.DataFrame()
    return numerical, textual


def align_domain(raw_dir: str | Path, domain: str, target_variable: str = "OT") -> tuple[pd.DataFrame, dict]:
    numerical, textual = load_raw_domain(raw_dir, domain)
    if "date" not in numerical:
        date_like = [c for c in numerical.columns if "date" in c.lower()]
        if not date_like:
            raise ValueError(f"{domain} has no date-like column")
        numerical = numerical.rename(columns={date_like[0]: "date"})
    numerical["date"] = pd.to_datetime(numerical["date"], errors="coerce")
    numerical = numerical.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if target_variable not in numerical:
        numeric_cols = numerical.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"{domain} has no numeric target candidate")
        target_variable = numeric_cols[0]

    drop_cols = [c for c in ["start_date", "end_date"] if c in numerical.columns]
    num_feature_cols = [
        c for c in numerical.columns
        if c not in {"date", *drop_cols} and pd.api.types.is_numeric_dtype(numerical[c])
    ]
    numerical[num_feature_cols] = (
        numerical[num_feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )

    if textual.empty:
        text_agg = pd.DataFrame({"date": numerical["date"], "text": "", "fact": "", "preds": ""})
    else:
        if "start_date" not in textual:
            text_date = [c for c in textual.columns if "date" in c.lower()]
            textual["start_date"] = textual[text_date[0]] if text_date else ""
        textual["date"] = pd.to_datetime(textual["start_date"], errors="coerce")
        textual["fact"] = textual.get("fact", "").map(clean_text)
        textual["preds"] = textual.get("preds", "").map(clean_text)
        textual["text"] = (textual["fact"].fillna("") + " " + textual["preds"].fillna("")).map(clean_text)
        text_agg = (
            textual.dropna(subset=["date"])
            .groupby("date", as_index=False)
            .agg(
                fact=("fact", lambda s: " ".join(x for x in s if x)[:8000]),
                preds=("preds", lambda s: " ".join(x for x in s if x)[:8000]),
                text=("text", lambda s: " ".join(x for x in s if x)[:12000]),
                text_rows=("text", "size"),
            )
        )
    merged = numerical.merge(text_agg, on="date", how="left")
    for col in ["fact", "preds", "text"]:
        merged[col] = merged[col].fillna("").map(clean_text)
    merged["text_rows"] = merged.get("text_rows", 0).fillna(0).astype(int)
    merged["has_text"] = merged["text"].str.len() > 0

    freq = infer_frequency(merged["date"])
    history_candidates, horizon_candidates = recommended_windows(freq)
    stats = {
        "domain": domain,
        "rows": int(len(merged)),
        "numeric_variables": int(len(num_feature_cols)),
        "target_variable": target_variable,
        "start_date": str(merged["date"].min().date()) if len(merged) else None,
        "end_date": str(merged["date"].max().date()) if len(merged) else None,
        "text_rows": int(merged["text_rows"].sum()),
        "text_coverage_ratio": float(merged["has_text"].mean()) if len(merged) else 0.0,
        "frequency": freq,
        "history_candidates": history_candidates,
        "horizon_candidates": horizon_candidates,
        "ot_mean": float(merged[target_variable].mean()),
        "ot_std": float(merged[target_variable].std(ddof=0)),
    }
    return merged, stats


def prepare_timemmd(raw_dir: str | Path, out_dir: str | Path, target_variable: str = "OT") -> dict:
    raw_dir = Path(raw_dir)
    out_dir = ensure_dir(out_dir)
    domains = discover_domains(raw_dir)
    all_stats = []
    for domain in domains:
        try:
            merged, stats = align_domain(raw_dir, domain, target_variable)
            merged.to_csv(out_dir / f"{domain}.csv", index=False)
            write_json(out_dir / f"{domain}_stats.json", stats)
            all_stats.append(stats)
        except Exception as exc:  # keep dataset scan resilient
            all_stats.append({"domain": domain, "error": str(exc)})
    summary = {"domains": domains, "stats": all_stats}
    write_json(out_dir / "summary.json", summary)
    return summary


def rank_domains(processed_dir: str | Path, max_domains: int = 3) -> list[str]:
    processed_dir = Path(processed_dir)
    stats = []
    for file in processed_dir.glob("*_stats.json"):
        item = pd.read_json(file, typ="series").to_dict()
        if "error" not in item:
            stats.append(item)
    stats.sort(key=lambda s: (s.get("text_coverage_ratio", 0), s.get("text_rows", 0), s.get("rows", 0)), reverse=True)
    return [s["domain"] for s in stats[:max_domains]]


def load_processed_domain(processed_dir: str | Path, domain: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(processed_dir) / f"{domain}.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df
