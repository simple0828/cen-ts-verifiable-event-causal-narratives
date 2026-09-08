from cen_ts.runtime.preflight import load_pretrained_gpt2_strict


def test_p1_gpt2_strict_loading_uses_pretrained_local_gpt2(local_gpt2) -> None:
    _, _, meta = load_pretrained_gpt2_strict(local_gpt2, pretrained=True, local_files_only=True)
    assert meta["model_class"] == "GPT2Model"
    assert meta["hidden_size"] == 768
    assert meta["random_init"] is False
