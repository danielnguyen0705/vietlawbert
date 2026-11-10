"""
tests/system/test/data_curation/test_inventory_team_corpus.py — Kiểm thử đơn vị cho kiểm kê kho văn bản.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

# Bỏ _common để tránh xung đột gói datasets
_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.inventory_team_corpus import char_ngrams, containment, scan_parquet_corpus

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier1

def test_char_ngrams():
    text = "Điều 1  Hôn nhân"
    ngrams = char_ngrams(text, 5)
    assert isinstance(ngrams, set)
    assert len(ngrams) > 0
    for item in ngrams:
        assert len(item) == 5
        assert item == item.lower()
        assert "  " not in item

    # Empty text -> empty set
    assert char_ngrams("", 5) == set()
    assert char_ngrams("   ", 5) == set()
    assert char_ngrams(None, 5) == set()
    assert char_ngrams("abc", 5) == set()

def test_containment():
    text = "Điều 1 Hôn nhân và gia đình"
    # Identical texts -> 1.0
    assert containment(text, text) == pytest.approx(1.0)
    assert containment(char_ngrams(text), char_ngrams(text)) == pytest.approx(1.0)

    # Disjoint -> 0.0
    disjoint = "XYZWQ1234567890"
    assert containment(text, disjoint) == pytest.approx(0.0)

    # Empty first argument -> 0.0 without division error
    assert containment("", text) == pytest.approx(0.0)
    assert containment(set(), char_ngrams(text)) == pytest.approx(0.0)

def test_tiny_fake_parquet_inventory(tmp_path: Path):
    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_number", pa.string()),
        ("doc_title", pa.string()),
        ("hierarchy_path", pa.string()),
        ("level", pa.string()),
        ("content", pa.string()),
        ("issue_date", pa.string()),
        ("effective_date", pa.string()),
        ("is_administrative", pa.bool_()),
    ])

    # File 1: 2 rows, doc1
    t1 = pa.Table.from_pydict({
        "chunk_id": ["c1", "c2"],
        "doc_id": ["doc1", "doc1"],
        "doc_number": ["01/2024/TT-BTP", "01/2024/TT-BTP"],
        "doc_title": ["Thông tư số 01", "Thông tư số 01"],
        "hierarchy_path": ["TT 01 > Điều 1", "TT 01 > Điều 2"],
        "level": ["ARTICLE", "ARTICLE"],
        "content": ["Nội dung điều 1 quy định chi tiết", "Nội dung điều 2 quy định chi tiết"],
        "issue_date": ["2024-01-01", "2024-01-01"],
        "effective_date": ["2024-02-01", "2024-02-01"],
        "is_administrative": [False, False],
    }, schema=schema)
    pq.write_table(t1, tmp_path / "part_001.parquet")

    # File 2: 1 row, doc2
    t2 = pa.Table.from_pydict({
        "chunk_id": ["c3"],
        "doc_id": ["doc2"],
        "doc_number": ["02/2024/ND-CP"],
        "doc_title": ["Nghị định số 02"],
        "hierarchy_path": ["ND 02 > Điều 1"],
        "level": ["ARTICLE"],
        "content": ["Nội dung điều 1 nghị định"],
        "issue_date": ["2024-03-01"],
        "effective_date": ["2024-04-01"],
        "is_administrative": [False],
    }, schema=schema)
    pq.write_table(t2, tmp_path / "part_002.parquet")

    res = scan_parquet_corpus(tmp_path)

    counts = res["counts"]
    assert counts["n_files"] == 2
    assert counts["total_rows_chunks"] == 3
    assert counts["n_distinct_doc_ids"] == 2
    assert counts["n_distinct_doc_numbers"] == 2
    assert counts["n_distinct_normalized_doc_numbers"] == 2
    assert res["schema_uniform_across_all_files"] is True