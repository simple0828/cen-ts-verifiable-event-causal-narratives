from types import SimpleNamespace

import pytest
import torch

from cen_tats.runtime.preflight import PreflightError, check_backbone
from cen_tats.runtime.tats_smoke import make_itransformer


def test_patchtst_is_rejected_for_v6() -> None:
    with pytest.raises(PreflightError):
        check_backbone({"training": {"backbone": "PatchTST"}})


def test_itransformer_forward_shape() -> None:
    check_backbone({"training": {"backbone": "iTransformer"}})
    model = make_itransformer(seq_len=8, pred_len=2, channels=13, d_model=16)
    x = torch.randn(2, 8, 13)
    dec = torch.zeros(2, 2, 13)
    y = model(x, None, dec, None)[:, -2:, :1]
    assert tuple(y.shape) == (2, 2, 1)
