from cen_tats.tats.official_adapter import TaTSForecastModel


def test_same_backbone_config_for_text_methods() -> None:
    a = TaTSForecastModel("PatchTST", seq_len=8, label_len=4, pred_len=2, text_dim=4, llm_dim=16, d_model=8, n_heads=2, d_ff=16)
    b = TaTSForecastModel("PatchTST", seq_len=8, label_len=4, pred_len=2, text_dim=4, llm_dim=16, d_model=8, n_heads=2, d_ff=16)
    assert a.backbone_name == b.backbone_name == "PatchTST"
    assert a.pred_len == b.pred_len == 2
