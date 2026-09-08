from __future__ import annotations

import sys
from types import SimpleNamespace

from cen_ts.utils.paths import TATS_ROOT


def forward_smoke() -> tuple[int, ...]:
    """Run one CPU iTransformer forward pass without GPT-2, training, or downloads."""
    import torch

    source_root = str(TATS_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from models.iTransformer import Model

    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=24, pred_len=48, output_attention=False,
        d_model=32, embed="timeF", freq="d", dropout=0.1, factor=1, n_heads=4,
        d_ff=64, e_layers=1, activation="gelu",
    )
    model = Model(config).eval()
    values = torch.randn(2, 24, 13)
    marks = torch.randn(2, 24, 4)
    with torch.no_grad():
        output = model(values, marks, None, None)
    expected = (2, 48, 13)
    if tuple(output.shape) != expected:
        raise AssertionError(f"Unexpected iTransformer output shape: {tuple(output.shape)}")
    return tuple(output.shape)
