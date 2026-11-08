"""
test_vlqa_ingest.py — Unit test cho logic nạp corpus VLQA và tạo dev set.

Phạm vi:
- iter_articles: tạo đúng chunk_id, doc_number, content; bỏ qua content rỗng/whitespace.
- check_unique_aids: raise ValueError khi trùng aid, qua khi duy nhất.
- main(--index vietlaw_sparse_idx): từ chối ghi đè index chính (exit code 2).
- main(--dry-run): chạy thành công trên fake corpus mà không cần ES.
- select_dev: kiểm tra tính tất định theo seed, số lượng N, và lọc bỏ case thiếu aid.
"""
from __future__ import annotations

import sys
import os
import json
import pytest

# Tránh tests/_common/datasets làm shadow package huggingface datasets khi pipeline/__init__ nạp sentence_transformers
_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from pipeline.ingest_vlqa_corpus import iter_articles, check_unique_aids, main
from tests._common.datasets.load_vlqa import select_dev
from configs.config import config

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier1


def test_iter_articles_builds_correct_fields_and_skips_empty():
    fake_corpus = [
        {
            "law_id": "12/2020/ND-CP",
            "content": [
                {"aid": 101, "content_Article": "Nội dung điều 101"},
                {"aid": 102, "content_Article": ""},
                {"aid": 103, "content_Article": "   	  "},
            ],
        },
        {
            "law_id": "99/2021/TT-BTC",
            "content": [
                {"aid": 201, "content_Article": "Nội dung điều 201"},
            ],
        },
    ]

    articles = list(iter_articles(fake_corpus))
    assert len(articles) == 2

    a0 = articles[0]
    assert a0["chunk_id"] == "vlqa_101"
    assert a0["aid"] == 101
    assert a0["doc_number"] == "12/2020/ND-CP"
    assert a0["content"] == "Nội dung điều 101"
    assert a0["macro_label"] == "VLQA"
    assert a0["is_effective"] is True
    assert "aid 101" in a0["hierarchy_path"]

    a1 = articles[1]
    assert a1["chunk_id"] == "vlqa_201"
    assert a1["aid"] == 201
    assert a1["doc_number"] == "99/2021/TT-BTC"
    assert a1["content"] == "Nội dung điều 201"


def test_check_unique_aids_raises_on_duplicates():
    # Trường hợp hợp lệ
    check_unique_aids([{"aid": 1}, {"aid": 2}, {"aid": 3}])

    # Trường hợp trùng lặp
    with pytest.raises(ValueError) as excinfo:
        check_unique_aids([{"aid": 1}, {"aid": 2}, {"aid": 1}, {"aid": 2}, {"aid": 4}])
    msg = str(excinfo.value)
    assert "aid bị trùng lặp" in msg
    assert "1" in msg


def test_main_refuses_vietlaw_sparse_idx_with_code_2():
    with pytest.raises(SystemExit) as excinfo:
        main(["--index", config.ES_INDEX_NAME])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo2:
        main(["--index", "vietlaw_sparse_idx"])
    assert excinfo2.value.code == 2


def test_dry_run_on_fake_corpus_succeeds_without_es(tmp_path):
    fake_corpus = [
        {
            "law_id": "01/2020/QH14",
            "content": [
                {"aid": 1, "content_Article": "Nội dung điều 1"},
                {"aid": 2, "content_Article": "Nội dung điều 2"},
            ],
        }
    ]
    corpus_file = tmp_path / "legal_corpus.json"
    corpus_file.write_text(json.dumps(fake_corpus, ensure_ascii=False), encoding="utf-8")

    # dry-run không được gọi ES, chạy thành công (return None, không raise)
    main(["--vlqa-dir", str(tmp_path), "--dry-run"])


def test_select_dev_properties():
    # 1. never includes a case without relevant aids
    mixed_cases = [
        {"qid": 1, "relevant_aids": [101]},
        {"qid": 2, "relevant_aids": []},
        {"qid": 3, "relevant_chunk_ids": ["vlqa_301"]},
        {"qid": 4, "relevant_aids": [], "relevant_chunk_ids": []},
        {"qid": 5, "relevant_aids": [501]},
    ]
    dev = select_dev(mixed_cases, n=10, seed=42)
    assert len(dev) == 3
    assert {c["qid"] for c in dev} == {1, 3, 5}

    # 2. returns exactly N
    many_cases = [{"qid": i, "relevant_aids": [i]} for i in range(1, 31)]
    assert len(select_dev(many_cases, n=7, seed=42)) == 7
    assert len(select_dev(many_cases, n=15, seed=42)) == 15

    # 3. same seed -> same qids (tính tất định)
    dev_a = select_dev(many_cases, n=10, seed=42)
    dev_b = select_dev(many_cases, n=10, seed=42)
    assert [c["qid"] for c in dev_a] == [c["qid"] for c in dev_b]

    # 4. different seed -> different order/selection
    dev_c = select_dev(many_cases, n=10, seed=99)
    assert [c["qid"] for c in dev_a] != [c["qid"] for c in dev_c]
