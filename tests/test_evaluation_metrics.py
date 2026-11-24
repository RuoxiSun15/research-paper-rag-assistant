import pytest

from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_uses_standard_k_denominator():
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 5) == pytest.approx(0.4)


def test_recall_at_k_counts_unique_relevant_results():
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)


def test_reciprocal_rank_finds_first_relevant_result():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"], 3) == pytest.approx(0.5)


def test_metrics_return_zero_when_nothing_is_relevant():
    assert precision_at_k(["a"], [], 1) == 0.0
    assert recall_at_k(["a"], [], 1) == 0.0
    assert reciprocal_rank(["a"], [], 1) == 0.0


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)
