"""
tests/system/test/retrieval_ir/test_rerank_topn_latency.py — Unit tests for top-N reranking and CPU latency helpers.
"""
from __future__ import annotations

import os
import sys
import re
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.rerank_topn import rerank_first_n, choose_n
from benchmark.measure_cpu_latency import rebuild_piece_texts, summarize, percentile

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_rerank_first_n_basic():
    candidates = ["a", "b", "c", "d"]
    scores = [0.1, 0.9, 0.5, 0.99]
    # Re-order first 2 candidates: "a" (0.1) and "b" (0.9) -> "b", "a". Followed by "c", "d"
    result = rerank_first_n(candidates, scores, 2)
    assert result == ["b", "a", "c", "d"]


def test_rerank_first_n_edge_cases():
    candidates = ["a", "b", "c"]
    scores = [0.3, 0.1, 0.2]
    # n <= 0 returns original list
    assert rerank_first_n(candidates, scores, 0) == ["a", "b", "c"]
    assert rerank_first_n(candidates, scores, -1) == ["a", "b", "c"]
    # n larger than list re-orders all candidates
    assert rerank_first_n(candidates, scores, 10) == ["a", "c", "b"]


def test_rerank_first_n_ties_keep_original_order():
    candidates = ["a", "b"]
    scores = [0.5, 0.5]
    result = rerank_first_n(candidates, scores, 2)
    assert result == ["a", "b"]

    candidates_rev = ["b", "a"]
    result_rev = rerank_first_n(candidates_rev, scores, 2)
    assert result_rev == ["b", "a"]


def test_choose_n_rule():
    fake_tune_rows = [
        {"n": 0, "NDCG@10": 0.6000},
        {"n": 5, "NDCG@10": 0.6500},
        {"n": 10, "NDCG@10": 0.6700},
        {"n": 20, "NDCG@10": 0.6760},
        {"n": 30, "NDCG@10": 0.6810},
        {"n": 50, "NDCG@10": 0.6850},
    ]
    # tol = 0.005 -> threshold = 0.6850 - 0.005 = 0.6800
    # rows >= 0.6800 are n=30 (0.6810) and n=50 (0.6850). Smallest eligible n is 30.
    assert choose_n(fake_tune_rows, tol=0.005) == 30

    # tol = 0.010 -> threshold = 0.6750 -> n=20 (0.6760)
    assert choose_n(fake_tune_rows, tol=0.010) == 20

    # tol = 0.001 -> threshold = 0.6840 -> n=50
    assert choose_n(fake_tune_rows, tol=0.001) == 50


class FakeWhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        matches = list(re.finditer(r"\S+", text))
        offsets = [m.span() for m in matches]
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


def test_rebuild_piece_texts_fake_tokenizer():
    tokenizer = FakeWhitespaceTokenizer()
    text = "alpha beta gamma delta epsilon zeta eta"
    articles = {"art_1": text}
    piece_meta = [
        {"piece_id": 0, "chunk_id": "art_1", "piece_index": 0, "n_pieces": 3, "n_tokens": 3},
        {"piece_id": 1, "chunk_id": "art_1", "piece_index": 1, "n_pieces": 3, "n_tokens": 3},
        {"piece_id": 2, "chunk_id": "art_1", "piece_index": 2, "n_pieces": 3, "n_tokens": 3},
    ]

    rebuilt = rebuild_piece_texts(articles, piece_meta, tokenizer, window=3, stride=2)
    assert len(rebuilt) == 3
    assert rebuilt[0] == "alpha beta gamma"
    assert rebuilt[1] == "gamma delta epsilon"
    assert rebuilt[2] == "epsilon zeta eta"

    bad_meta = [
        {"piece_id": 0, "chunk_id": "art_1", "piece_index": 0, "n_pieces": 3, "n_tokens": 999},
        {"piece_id": 1, "chunk_id": "art_1", "piece_index": 1, "n_pieces": 3, "n_tokens": 3},
        {"piece_id": 2, "chunk_id": "art_1", "piece_index": 2, "n_pieces": 3, "n_tokens": 3},
    ]
    with pytest.raises((AssertionError, ValueError)):
        rebuild_piece_texts(articles, bad_meta, tokenizer, window=3, stride=2)


def test_summarize_share_under_500ms():
    res = summarize([100.0, 200.0, 600.0])
    assert res["share_under_500ms"] == pytest.approx(2 / 3)
    assert res["P50_ms"] == pytest.approx(200.0)
    assert res["P95_ms"] == pytest.approx(560.0)