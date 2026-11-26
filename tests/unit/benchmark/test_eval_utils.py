# -*- coding: utf-8 -*-
"""Self-check cho eval_utils: Min-Max BM25 (B2), MRR@10 chọn checkpoint (A2), pool hard-negative (C1).
Thuần numpy, không kéo torch/sbert."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from eval_utils import minmax_norm, mrr_at_k, build_candidate_pool  # noqa: E402


def test_minmax_norm():
    assert minmax_norm([]) == []
    assert minmax_norm([5.0]) == [0.0]                 # 1 phần tử -> 0
    assert minmax_norm([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    assert minmax_norm([2.0, 2.0]) == [0.0, 0.0]       # mọi giá trị bằng nhau -> 0 (không chia 0)


def test_mrr_at_k():
    assert mrr_at_k(np.eye(3), k=10) == 1.0             # gold ở đường chéo, hạng 1
    assert mrr_at_k(np.array([[0.0, 1.0], [1.0, 0.0]]), k=10) == 0.5   # gold luôn ở hạng 2
    assert mrr_at_k(np.array([[0.0, 1.0], [1.0, 0.0]]), k=1) == 0.0    # cắt ở k=1 -> trượt
    assert mrr_at_k(np.empty((0, 0)), k=10) == 0.0


class _FakeES:
    def search(self, q, top_k=10):
        return [{"chunk_id": "hn1", "content": "noi dung hard negative du dai", "doc_number": "X", "article": ""}]


def test_build_candidate_pool_them_hard_negative_va_dedup():
    samples = [{"ground_truth_chunk": "g1", "raw_content": "noi dung gold",
                "query": "cau hoi", "ground_truth_doc_number": "A"}]
    pool = build_candidate_pool(samples, es=_FakeES(), per_query_hard=5)
    ids = {c["chunk_id"] for c in pool}
    assert "g1" in ids and "hn1" in ids                # có cả gold lẫn hard-negative

    pool2 = build_candidate_pool(samples * 2, es=_FakeES())
    assert len(pool2) == 2                             # trùng gold + trùng hard-negative -> dedup còn 2


def test_build_candidate_pool_khong_es_chi_gold():
    samples = [{"ground_truth_chunk": "g1", "raw_content": "noi dung gold", "query": "q"}]
    pool = build_candidate_pool(samples, es=None)
    ids = {c["chunk_id"] for c in pool}
    assert ids == {"g1"}                               # không ES -> chỉ còn gold
