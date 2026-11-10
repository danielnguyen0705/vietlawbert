"""Tier-1: Cohen's kappa đúng số (ví dụ tính tay)."""
from tests._common.datasets.compute_kappa import cohen_kappa


def test_perfect_agreement():
    assert cohen_kappa(["S", "S", "B", "B"], ["S", "S", "B", "B"]) == 1.0

def test_known_value_half():
    # a=[S,S,S,B], b=[S,S,B,B]: Po=0.75, Pe=0.5 → κ=0.5
    k = cohen_kappa(["S", "S", "S", "B"], ["S", "S", "B", "B"])
    assert abs(k - 0.5) < 1e-9

def test_all_same_label_guard():
    assert cohen_kappa(["S", "S", "S"], ["S", "S", "S"]) == 1.0
