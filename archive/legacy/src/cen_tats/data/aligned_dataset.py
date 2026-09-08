from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AlignedDataset:
    name: str
    frame: pd.DataFrame
    target: str
    text_col: str = "aux_text"
    date_col: str = "date"

    @property
    def n_timepoints(self) -> int:
        return int(len(self.frame))

    @property
    def text_coverage(self) -> float:
        text = self.frame[self.text_col].fillna("").astype(str).str.strip()
        return float((text != "").mean()) if len(text) else 0.0
