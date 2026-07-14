# P3A Environment Event Extraction Input Audit

- Rows: 15248
- Official train / validation / test: `0..10673` / `10673..12199` / `12199..15248`
- P3A train_core / prompt_dev: `(0, 9072)` / `(9072, 10673)`
- Non-empty text rate: 0.984654
- Raw / normalized unique non-empty texts: 2080 / 2080
- Exact duplicate rate: 0.861463
- Average characters / GPT-2 tokens: 152.22 / 28.847275875849206
- Max characters / GPT-2 tokens: 895 / 165
- Date / multiple-date rates: 0.273611 / 0.026109
- Forecast-language rate: 0.068070
- Multiple-source heuristic rate: 0.055948
- GPT-2 truncation-risk texts (>1800 tokens): 0
- Target semantics: resolved as daily Air Quality Index (AQI), from local Time-MMD materials.

Normalization is used only for deduplication and hashing. LLM requests retain original text, dates, numbers, entities, and casing.
