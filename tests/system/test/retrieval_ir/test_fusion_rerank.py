"""
test_fusion_rerank.py — Kiểm thử các hàm Weighted RRF, quy tắc chọn tham số và bộ nạp tune.
"""
from __future__ import annotations

import sys
import os
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from benchmark.tune_vlqa_fusion import (
    weighted_rrf,
    order_rerank_candidates,
    select_best_fusion_params,
    load_tune_records,
)
from benchmark.evaluate_vlqa_article import rrf_fuse

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_weighted_rrf_hand_computed_example():
    rankings = [["a", "b"], ["b", "c"]]
    weights = [0.25, 0.75]
    result = weighted_rrf(rankings, weights, k=1)
    assert result == ["b", "c", "a"]


def test_weighted_rrf_equal_weights_matches_rrf_fuse():
    rankings = [
        ["art_10", "art_20", "art_30", "art_40"],
        ["art_20", "art_50", "art_10", "art_60"],
    ]
    weights = [0.5, 0.5]
    k = 60

    w_res = weighted_rrf(rankings, weights, k=k, depth=10)
    std_res = rrf_fuse(rankings, k=k, depth=10)
    assert w_res == std_res


def test_weighted_rrf_tie_break_by_chunk_id():
    rankings = [["y"], ["x"]]
    weights = [0.5, 0.5]
    result = weighted_rrf(rankings, weights, k=60)
    assert result == ["x", "y"]


def test_select_best_fusion_params_tie_rules():
    # 1. NDCG rõ ràng (chênh lệch >= 0.0005)
    grid_clear = [
        {"w_dense": 0.5, "w_bm25": 0.5, "k": 10, "NDCG@10": 0.6000, "F2@2": 0.50},
        {"w_dense": 0.6, "w_bm25": 0.4, "k": 20, "NDCG@10": 0.6010, "F2@2": 0.40},
    ]
    assert select_best_fusion_params(grid_clear)["w_dense"] == 0.6

    # 2. Hòa NDCG (< 0.0005) -> Chọn F2@2 cao hơn
    grid_tie_f2 = [
        {"w_dense": 0.5, "w_bm25": 0.5, "k": 10, "NDCG@10": 0.6000, "F2@2": 0.52},
        {"w_dense": 0.6, "w_bm25": 0.4, "k": 20, "NDCG@10": 0.6003, "F2@2": 0.55},
    ]
    assert select_best_fusion_params(grid_tie_f2)["w_dense"] == 0.6

    # 3. Hòa NDCG & F2@2 -> Chọn k nhỏ hơn
    grid_tie_k = [
        {"w_dense": 0.7, "w_bm25": 0.3, "k": 60, "NDCG@10": 0.6002, "F2@2": 0.55},
        {"w_dense": 0.7, "w_bm25": 0.3, "k": 20, "NDCG@10": 0.6003, "F2@2": 0.55},
    ]
    assert select_best_fusion_params(grid_tie_k)["k"] == 20

    # 4. Hòa NDCG, F2@2 & k -> Chọn w_dense nhỏ hơn
    grid_tie_w = [
        {"w_dense": 0.8, "w_bm25": 0.2, "k": 20, "NDCG@10": 0.6003, "F2@2": 0.55},
        {"w_dense": 0.6, "w_bm25": 0.4, "k": 20, "NDCG@10": 0.6003, "F2@2": 0.55},
    ]
    assert select_best_fusion_params(grid_tie_w)["w_dense"] == 0.6


def test_order_rerank_candidates():
    candidates = ["a", "b", "c"]
    scores = [0.1, 0.9, 0.9]
    assert order_rerank_candidates(candidates, scores) == ["b", "c", "a"]


def test_load_tune_records_raises_on_dev_record(tmp_path):
    # Dữ liệu chứa bản ghi split == "dev"
    bad_records = [
        {"split": "tune", "qid": 1001, "ranked": []},
        {"split": "dev", "qid": 1002, "ranked": []},
    ]

    with pytest.raises(ValueError) as excinfo:
        load_tune_records(bad_records)
    assert "CẤM: Phát hiện bản ghi dev" in str(excinfo.value)

    # Thử qua đường dẫn file
    bad_file = tmp_path / "bad_tune.jsonl"
    import json
    with open(bad_file, "w", encoding="utf-8") as f:
        for r in bad_records:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(ValueError):
        load_tune_records(bad_file)