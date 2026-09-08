import torch

from cen_tats.tats.official_adapter import TaTSForecastModel


def test_tats_input_shape() -> None:
    model = TaTSForecastModel("PatchTST", seq_len=8, label_len=4, pred_len=2, text_dim=4, llm_dim=16, d_model=8, n_heads=2, d_ff=16)
    y = model(torch.randn(3, 8, 1), torch.randn(3, 8, 16))
    assert tuple(y.shape) == (3, 2, 1)
