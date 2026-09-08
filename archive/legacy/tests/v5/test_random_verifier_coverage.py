from cen_tats.evaluation.verifier_metrics import corruption_benchmark_scores


def test_random_verifier_coverage() -> None:
    metrics = corruption_benchmark_scores([0.8, 0.7], [0.2, 0.1], threshold=0.5)
    assert metrics["wrong_keep_rate"] == 0.0
    assert metrics["wrong_delete_rate"] == 0.0
