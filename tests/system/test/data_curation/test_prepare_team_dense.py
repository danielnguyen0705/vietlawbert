"""
tests/system/test/data_curation/test_prepare_team_dense.py — Kiểm thử đơn vị cho việc chuẩn bị dữ liệu dense trên kho nhóm.
"""
from __future__ import annotations

import os
import sys
import pytest
import pyarrow as pa

# Bỏ _common để tránh xung đột gói datasets
_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.prepare_team_dense import (
    check_duplicates,
    compute_dedup_drops,
    apply_dedup_to_table,
)

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier1


def test_dedup_rule_last_occurrence_wins():
    """Kiểm thử quy tắc de-duplicate: bản ghi xuất hiện sau cùng sẽ được giữ lại, tổng số hàng chính xác."""
    files_with_cids = [
        ("file_0.parquet", ["c1", "c2", "c1"]),
        ("file_1.parquet", ["c2", "c3"]),
    ]

    drops = compute_dedup_drops(files_with_cids)

    # c1 xuất hiện ở file 0 hàng 0 và hàng 2 -> hàng 2 là cuối cùng -> drop (0, 0)
    # c2 xuất hiện ở file 0 hàng 1 và file 1 hàng 0 -> file 1 hàng 0 là cuối cùng -> drop (0, 1)
    # c3 xuất hiện ở file 1 hàng 1 -> giữ
    assert drops == {(0, 0), (0, 1)}

    # Áp dụng lên bảng PyArrow
    t0 = pa.Table.from_pydict({
        "chunk_id": ["c1", "c2", "c1"],
        "level": ["ARTICLE", "ARTICLE", "ARTICLE"],
        "content": ["nội dung cũ 1", "nội dung cũ 2", "nội dung mới 1"],
    })
    t1 = pa.Table.from_pydict({
        "chunk_id": ["c2", "c3"],
        "level": ["ARTICLE", "ARTICLE"],
        "content": ["nội dung mới 2", "nội dung 3"],
    })

    t0_slim = apply_dedup_to_table(t0, {r for f, r in drops if f == 0})
    t1_slim = apply_dedup_to_table(t1, {r for f, r in drops if f == 1})

    assert t0_slim.num_rows == 1
    assert t0_slim["chunk_id"].to_pylist() == ["c1"]
    assert t0_slim["content"].to_pylist() == ["nội dung mới 1"]

    assert t1_slim.num_rows == 2
    assert t1_slim["chunk_id"].to_pylist() == ["c2", "c3"]
    assert t1_slim["content"].to_pylist() == ["nội dung mới 2", "nội dung 3"]

    total_rows = t0_slim.num_rows + t1_slim.num_rows
    assert total_rows == 3


def test_dupcheck_counts_identical_and_different_content():
    """Kiểm thử phát hiện và phân loại trùng lặp: giống nhau vs khác nhau về nội dung và số hiệu văn bản."""
    records = [
        # c1 lặp lại 2 lần với nội dung giống hệt
        {"chunk_id": "c1", "doc_number": "01/2026/QD", "content": "cùng nội dung", "file_name": "f1", "row_idx": 0},
        {"chunk_id": "c1", "doc_number": "01/2026/QD", "content": "cùng nội dung", "file_name": "f2", "row_idx": 0},
        # c2 lặp lại 2 lần với nội dung khác nhau, cùng doc_number
        {"chunk_id": "c2", "doc_number": "02/2026/QD", "content": "bản a", "file_name": "f1", "row_idx": 1},
        {"chunk_id": "c2", "doc_number": "02/2026/QD", "content": "bản b", "file_name": "f2", "row_idx": 1},
        # c3 lặp lại 2 lần với nội dung khác nhau, khác doc_number
        {"chunk_id": "c3", "doc_number": "03/2026/QD", "content": "văn bản 1", "file_name": "f1", "row_idx": 2},
        {"chunk_id": "c3", "doc_number": "04/2026/QD", "content": "văn bản 2", "file_name": "f2", "row_idx": 2},
        # c4 không lặp
        {"chunk_id": "c4", "doc_number": "05/2026/QD", "content": "duy nhất", "file_name": "f1", "row_idx": 3},
    ]

    res = check_duplicates(records)
    assert res["repeated_ids_count"] == 3
    assert res["identical_content_count"] == 1
    assert res["different_content_count"] == 2
    assert res["different_content_same_doc_count"] == 1
    assert res["different_content_diff_doc_count"] == 1
