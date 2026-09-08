from cen_ts.runtime.preflight import check_cuda


def test_cuda_available_and_gpt2_forward(local_gpt2, cuda_device) -> None:
    result = check_cuda(local_gpt2, gpu=0)
    assert result["cuda_available"] is True
    assert result["device_count"] >= 1
    assert "NVIDIA" in result["gpu_name"]
    assert result["gpu_forward_ok"] is True
