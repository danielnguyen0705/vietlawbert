"""
tests/system/test/retrieval_ir/test_team_eval.py — Kiểm thử đơn vị cho việc đánh giá trên kho văn bản nhóm.
"""
from __future__ import annotations

import os
import sys
import subprocess
import pytest

# Bỏ _common để tránh xung đột gói datasets
_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.metrics_article import hit_at_k, recall_at_k

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4

def to_group_ranking(ranked_chunk_ids: list[str], chunk_to_group: dict[str, str]) -> list[str]:
    """Hàm thuần túy ánh xạ chunk sang group id, giữ thứ tự và bỏ trùng."""
    seen = set()
    res = []
    for cid in ranked_chunk_ids:
        gid = chunk_to_group.get(cid, cid)
        if gid not in seen:
            seen.add(gid)
            res.append(gid)
    return res

def test_to_group_ranking():
    ranked = ["c1", "c2", "x", "c3"]
    mapping = {"c1": "g1", "c2": "g1", "c3": "g2"}
    expected = ["g1", "x", "g2"]
    assert to_group_ranking(ranked, mapping) == expected

def test_group_recall_and_hit():
    # Câu hỏi có gold groups {g1, g2}, chỉ lấy được bản sao của g1 ở hạng 1
    chunk_to_group = {"c1_copy1": "g1", "c1_copy2": "g1", "c2": "g2"}
    gold_groups = {"g1", "g2"}

    retrieved_cids = ["c1_copy1", "other_chunk", "another_chunk"]
    ranked_groups = to_group_ranking(retrieved_cids, chunk_to_group)

    rec10 = recall_at_k(ranked_groups, gold_groups, 10)
    hit1 = hit_at_k(ranked_groups, gold_groups, 1)

    assert rec10 == pytest.approx(0.5)
    assert hit1 == pytest.approx(1.0)

def test_aid_with_no_gold_members_counted_in_denominator():
    # aid 1 có gold chunk c1 -> g1; aid 2 không tìm thấy văn bản -> g2 không có chunk nào
    chunk_to_group = {"c1": "aid:1"}
    gold_groups = {"aid:1", "aid:2"}

    retrieved_cids = ["c1", "c_unrelated"]
    ranked_groups = to_group_ranking(retrieved_cids, chunk_to_group)

    rec10 = recall_at_k(ranked_groups, gold_groups, 10)
    assert rec10 == pytest.approx(0.5)

def test_ingest_team_seg_guard_exit_2():
    # Chạy pipeline.ingest_team_seg với --index khác team_sparse_seg_idx phải thoát mã 2
    cmd = [
        sys.executable,
        "-m",
        "pipeline.ingest_team_seg",
        "--index",
        "vlqa_sparse_seg_idx",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2