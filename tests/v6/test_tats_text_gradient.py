from cen_ts.runtime.tats_smoke import run_itransformer_smoke


def test_tats_text_projection_gradient_and_text_sensitivity(local_gpt2, cuda_device) -> None:
    result = run_itransformer_smoke(model_path=local_gpt2, device=cuda_device)
    assert result["projection_gradient_norm"] > 1e-8
    assert result["prediction_real_vs_zero_max_diff"] > 1e-6
    assert result["prediction_real_vs_shuffle_max_diff"] > 1e-6
    assert result["output_shape"] == [2, 2, 1]
