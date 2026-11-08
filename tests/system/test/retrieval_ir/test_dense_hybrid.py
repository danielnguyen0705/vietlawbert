"""
test_dense_hybrid.py — Kiểm thử các hàm tổng hợp Dense, dung hợp Hybrid RRF và nạp vector Qdrant.
"""
from __future__ import annotations

import sys
import os
import json
import hashlib
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from benchmark.evaluate_vlqa_article import aggregate_pieces, rrf_fuse
from pipeline.ingest_vlqa_dense import main as ingest_main, verify_checksums

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_aggregate_pieces_max_score_first_occurrence():
    hits = [("a", 0.9), ("b", 0.8), ("a", 0.7), ("c", 0.6)]
    assert aggregate_pieces(hits) == ["a", "b", "c"]


def test_rrf_fuse_hand_computed_order():
    rankings = [["a", "b", "c"], ["b", "c", "d"]]
    order = rrf_fuse(rankings, k=60, depth=10)
    assert order == ["b", "c", "a", "d"]


def test_rrf_fuse_tie_break_lexicographical():
    assert rrf_fuse([["x"], ["y"]], k=60) == ["x", "y"]


def test_ingest_vlqa_dense_guard_exits_code_2():
    with pytest.raises(SystemExit) as excinfo:
        ingest_main(["--collection", "vietlawbert_chunks"])
    assert excinfo.value.code == 2


def test_verify_checksums_detects_changed_or_missing_file(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"

    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")

    h1 = hashlib.sha256(b"hello").hexdigest()
    h2 = hashlib.sha256(b"world").hexdigest()

    manifest = {
        "sha256": {
            "file1.txt": h1,
            "file2.txt": h2,
        }
    }

    # Ban đầu không có file lỗi
    assert verify_checksums(tmp_path, manifest) == []

    # Sửa đổi nội dung file2
    f2.write_text("modified content", encoding="utf-8")
    assert verify_checksums(tmp_path, manifest) == ["file2.txt"]

    # Xóa file1
    f1.unlink()
    assert sorted(verify_checksums(tmp_path, manifest)) == ["file1.txt", "file2.txt"]