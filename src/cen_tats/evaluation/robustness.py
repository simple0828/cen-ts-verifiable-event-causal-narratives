from __future__ import annotations

import random


def shuffled_texts(texts: list[str], seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    out = list(texts)
    rng.shuffle(out)
    return out
