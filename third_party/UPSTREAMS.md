# Upstreams

`third_party/tats/` is the only source copy retained in this repository. It is based
on the pinned TaTS commit below and contains the minimal text-mode, strict local GPT-2,
and run-isolation adaptations needed by this project. The upstream commit and model
SHA-256 values are recorded in `third_party/tats/UPSTREAM.json`; the upstream license
is retained unchanged as `third_party/tats/LICENSE.txt`.

| Project | Repository | Commit | License | Retained purpose | Local changes |
|---|---|---|---|---|---|
| TaTS | https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS | a053503674c61c54d101d01d47c9d680288a7c9a | see `third_party/tats/LICENSE.txt` | iTransformer backbone and text-channel training | data loading, strict local GPT-2, CLI/output isolation; backbone unchanged |
