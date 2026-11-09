"""
test_export_candidates.py — Kiểm thử tập tune VLQA và xuất danh sách ứng viên BM25.
"""
from __future__ import annotations

import sys
import os
import pytest
from pathlib import Path

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from tests._common.datasets.load_vlqa import split_tune
from benchmark.export_vlqa_candidates import (
    export_rankings,
    main as export_main,
)
from configs.paths import ROOT_DIR

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_split_tune_filters_and_sorts():
    cases = [
        {"qid": 100, "relevant_aids": [1], "relevant_chunk_ids": ["vlqa_1"]},
        {"qid": 50, "relevant_aids": [2], "relevant_chunk_ids": ["vlqa_2"]},
        {"qid": 200, "relevant_aids": [], "relevant_chunk_ids": []},  # Không có aid -> loại bỏ
        {"qid": 10, "relevant_aids": [3], "relevant_chunk_ids": ["vlqa_3"]},
        {"qid": 30, "relevant_aids": [4], "relevant_chunk_ids": ["vlqa_4"]},
    ]
    # Dev chứa qid 50 và 10
    dev_qids = {50, 10}
    tune = split_tune(cases, dev_qids)

    # Chỉ giữ lại qid 30 và 100, sắp xếp theo qid tăng dần
    assert len(tune) == 2
    assert [c["qid"] for c in tune] == [30, 100]
    assert 200 not in [c["qid"] for c in tune]
    assert 50 not in [c["qid"] for c in tune]
    assert 10 not in [c["qid"] for c in tune]


def test_export_rankings_with_fake_search_fn():
    fake_search = lambda q: [("vlqa_1", 3.0), ("vlqa_1", 2.0), ("vlqa_2", 1.0)]
    questions = [{"qid": 999, "query": "quy định pháp luật"}]

    records = export_rankings(questions, split="dev", search_fn=fake_search)
    assert len(records) == 1
    rec = records[0]
    assert rec["split"] == "dev"
    assert rec["qid"] == 999
    assert rec["ranked"] == ["vlqa_1", "vlqa_2"]
    assert rec["scores"] == [3.0, 1.0]


def test_export_main_guard_refuses_main_es_index():
    with pytest.raises(SystemExit) as excinfo:
        export_main(["--seg-index", "vietlaw_sparse_idx"])
    assert excinfo.value.code == 2


def test_export_main_guard_refuses_out_dir_inside_repo(tmp_path):
    inside_dir = ROOT_DIR / "benchmark" / "temp_inside_export"
    with pytest.raises(SystemExit) as excinfo:
        export_main(["--out-dir", str(inside_dir)])
    assert excinfo.value.code == 2