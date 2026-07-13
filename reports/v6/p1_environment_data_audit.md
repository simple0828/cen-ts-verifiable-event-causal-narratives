# P1 Environment Data Audit

- Dataset: Environment
- CSV path: `C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\third_party\TaTS\data\Environment.csv`
- Data path source: `adapter_third_party_tats_data`
- Total time points: 15248
- Time range: 1982-01-01 00:00:00 to 2023-09-30 00:00:00
- Frequency mode: 1 days 00:00:00
- Target: `OT`
- Numeric missing rate: 0.000000
- Text columns: fact, preds
- Text coverage rate: 0.999082
- Empty text count: 14
- Duplicate text count: 13088
- Average / max text length: 484.92 / 1558
- Constructible windows: 15035
- Split hash: `b80f7a6c5c4ed081e9e7b8255f006a4ead33913ab6c38fb0e8ef9bc2563e6178`

| split | point range [start,end) | windows |
| --- | --- | --- |
| train | [0, 10673] | 10602 |
| model_val | [10673, 12199] | 1455 |
| test | [12199, 15248] | 2978 |

Hard checks: no rows are dropped for missing text; splits are chronological; scaler is fit on train only; test is not used for parameter selection; M0/M1/Z0/S0 share the same split.
