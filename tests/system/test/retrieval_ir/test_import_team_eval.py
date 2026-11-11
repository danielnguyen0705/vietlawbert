"""
tests/system/test/retrieval_ir/test_import_team_eval.py — Kiểm thử đơn vị cho Task 9E (import kết quả & ablation legacy).
"""
from __future__ import annotations

import os
import sys
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.import_team_eval_9d import (
    order_rerank_candidates,
    verify_rrf_candidates,
)
from benchmark.evaluate_vlqa_article import rrf_fuse
from pipeline.ingest_team_legacy import main as ingest_legacy_main

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_rerank_ordering_preserves_ties():
    candidates = ["a", "b", "c"]
    scores = [0.2, 0.9, 0.9]
    assert order_rerank_candidates(candidates, scores) == ["b", "c", "a"]


def test_local_rrf_vs_given_candidates_match_and_mismatch():
    b1 = ["c1", "c2", "c3"]
    d1 = ["c2", "c3", "c4"]
    expected_q1 = rrf_fuse([b1, d1], k=60, depth=100)[:2]

    b2 = ["x1", "x2"]
    d2 = ["x2", "x3"]
    wrong_q2 = ["x3", "x1"]

    bm25_map = {("vlqa", 1): b1, ("vlqa", 2): b2}
    dense_map = {("vlqa", 1): d1, ("vlqa", 2): d2}

    # Trường hợp khớp hoàn toàn
    hybrid_ok, mismatches_ok = verify_rrf_candidates(
        {("vlqa", 1): b1},
        {("vlqa", 1): d1},
        {("vlqa", 1): expected_q1},
        k=60,
        depth=100,
        top_n=2,
    )
    assert mismatches_ok == []
    assert ("vlqa", 1) in hybrid_ok

    # Trường hợp câu hỏi 2 bị lệch
    _, mismatches_bad = verify_rrf_candidates(
        bm25_map,
        dense_map,
        {("vlqa", 1): expected_q1, ("vlqa", 2): wrong_q2},
        k=60,
        depth=100,
        top_n=2,
    )
    assert mismatches_bad == [("vlqa", 2)]


def test_ingest_team_legacy_guard_exits_code_2():
    with pytest.raises(SystemExit) as excinfo:
        ingest_legacy_main(["--index", "team_sparse_seg_idx"])
    assert excinfo.value.code == 2
