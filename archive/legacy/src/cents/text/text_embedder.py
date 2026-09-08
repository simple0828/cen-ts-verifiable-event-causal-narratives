from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfSvdEmbedder:
    n_components: int = 16
    max_features: int = 2048

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), min_df=1)
        x = self.vectorizer.fit_transform(texts)
        n_components = max(1, min(self.n_components, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=2026)
        arr = self.svd.fit_transform(x)
        if arr.shape[1] < self.n_components:
            arr = np.pad(arr, ((0, 0), (0, self.n_components - arr.shape[1])))
        return arr.astype(float)

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        arr = self.svd.transform(x)
        if arr.shape[1] < self.n_components:
            arr = np.pad(arr, ((0, 0), (0, self.n_components - arr.shape[1])))
        return arr.astype(float)


def fit_text_embeddings(df: pd.DataFrame, cache_path: str | Path | None = None, n_components: int = 16) -> np.ndarray:
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return np.load(cache_path)
    texts = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
    embedder = TfidfSvdEmbedder(n_components=n_components)
    arr = embedder.fit_transform(texts)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr

