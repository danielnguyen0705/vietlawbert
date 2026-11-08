"""
test_metrics_article.py — Unit test cho các hàm đo IR cấp điều luật (benchmark/metrics_article.py).

Kiểm tra bằng các ví dụ tính tay:
- ranked = ["a","x","b","y"], gold = {"a","b"}
- ranked = ["x","y","a"], gold = {"a"}
- Trường hợp biên: gold rỗng, không tìm thấy, chia cho 0
- Hàm aggregate tính macro average chuẩn xác
"""
from __future__ import annotations

import math
import pytest

import sys
import os

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from benchmark.metrics_article import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    ap_at_k,
    f2_at_k,
    aggregate,
)


for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_hand_crafted_example_one():
    ranked = ["a", "x", "b", "y"]
    gold = {"a", "b"}

    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(1.0)
    assert precision_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert mrr_at_k(ranked, gold, 10) == pytest.approx(1.0)

    # ndcg@3 = (1 + 1/log2(4)) / (1 + 1/log2(3))
    expected_ndcg3 = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(expected_ndcg3)

    # ap@3 = (1/1 + 2/3) / 2
    expected_ap3 = (1.0 / 1.0 + 2.0 / 3.0) / 2.0
    assert ap_at_k(ranked, gold, 3) == pytest.approx(expected_ap3)

    # f2@2: P=0.5, R=0.5 -> 5*0.5*0.5 / (4*0.5 + 0.5) = 1.25 / 2.5 = 0.5
    assert f2_at_k(ranked, gold, 2) == pytest.approx(0.5)


def test_hand_crafted_example_two():
    ranked = ["x", "y", "a"]
    gold = {"a"}

    assert mrr_at_k(ranked, gold, 10) == pytest.approx(1.0 / 3.0)
    assert hit_at_k(ranked, gold, 2) == pytest.approx(0.0)
    assert hit_at_k(ranked, gold, 3) == pytest.approx(1.0)


def test_edge_cases_empty_gold_and_no_match():
    # empty gold -> all metrics 0.0
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, set(), 3) == 0.0
    assert recall_at_k(ranked, set(), 3) == 0.0
    assert precision_at_k(ranked, set(), 3) == 0.0
    assert mrr_at_k(ranked, set(), 3) == 0.0
    assert ndcg_at_k(ranked, set(), 3) == 0.0
    assert ap_at_k(ranked, set(), 3) == 0.0
    assert f2_at_k(ranked, set(), 3) == 0.0

    # no relevant in ranked -> f2 = 0.0 (không vướng lỗi ZeroDivisionError)
    gold = {"z"}
    assert hit_at_k(ranked, gold, 3) == 0.0
    assert recall_at_k(ranked, gold, 3) == 0.0
    assert precision_at_k(ranked, gold, 3) == 0.0
    assert mrr_at_k(ranked, gold, 3) == 0.0
    assert ndcg_at_k(ranked, gold, 3) == 0.0
    assert ap_at_k(ranked, gold, 3) == 0.0
    assert f2_at_k(ranked, gold, 3) == 0.0


def test_aggregate_macro_average():
    per_query = [
        {"recall@1": 1.0, "mrr@10": 1.0, "f2@5": 0.8},
        {"recall@1": 0.0, "mrr@10": 0.5, "f2@5": 0.4},
    ]
    avg = aggregate(per_query)
    assert avg["recall@1"] == pytest.approx(0.5)
    assert avg["mrr@10"] == pytest.approx(0.75)
    assert avg["f2@5"] == pytest.approx(0.6)
