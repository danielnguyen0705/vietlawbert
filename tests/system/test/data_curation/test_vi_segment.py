"""
test_vi_segment.py — Kiểm thử tiền xử lý tách từ tiếng Việt và lập chỉ mục giữ dấu.

Phạm vi:
- segment_text: xử lý chuẩn xác khoảng trắng, xâu rỗng, áp dụng hàm tách từ.
- get_segmenter: kiểm tra raise ValueError khi truyền tên thư viện lạ.
- ingest_vlqa_seg guards: từ chối các tên index khác vlqa_sparse_seg_idx (exit code 2).
- ES analyzer test: kiểm tra vi_keep_accents giữ dấu và giữ nguyên từ ghép có dấu gạch dưới.
"""
from __future__ import annotations

import sys
import os
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == '_common']
for p in _removed:
    sys.path.remove(p)

from preprocess.vi_segment import get_segmenter, segment_text
from pipeline.ingest_vlqa_seg import main as ingest_main
from configs.config import config

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier1


def test_segment_text_with_fake_fn():
    fake_seg = lambda s: s.replace("hôn nhân", "hôn_nhân").replace("gia đình", "gia_đình")

    # Xâu rỗng hoặc chỉ khoảng trắng
    assert segment_text("", fake_seg) == ""
    assert segment_text("   \t\n  ", fake_seg) == ""

    # Thu gọn khoảng trắng dư thừa và tách từ đúng
    raw = "  Quy định   về hôn nhân   và\t gia đình.  "
    expected = "Quy định về hôn_nhân và gia_đình."
    assert segment_text(raw, fake_seg) == expected


def test_get_segmenter_unknown_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        get_segmenter("unknown_segmenter_xyz")
    assert "Không hỗ trợ segmenter" in str(excinfo.value)


def test_ingest_vlqa_seg_guard_protects_other_indices():
    with pytest.raises(SystemExit) as exc1:
        ingest_main(["--index", "vlqa_sparse_idx"])
    assert exc1.value.code == 2

    with pytest.raises(SystemExit) as exc2:
        ingest_main(["--index", "vietlaw_sparse_idx"])
    assert exc2.value.code == 2


@pytest.mark.requires_es
def test_es_vi_keep_accents_analyzer():
    from elasticsearch import Elasticsearch

    es = Elasticsearch(config.ES_HOST, request_timeout=10.0)
    if not es.ping():
        pytest.skip("Elasticsearch không khả dụng tại " + str(config.ES_HOST))

    if not es.indices.exists(index="vlqa_sparse_seg_idx"):
        pytest.skip("Index vlqa_sparse_seg_idx chưa được tạo, bỏ qua kiểm tra analyzer.")

    resp = es.indices.analyze(
        index="vlqa_sparse_seg_idx",
        analyzer="vi_keep_accents",
        text="Hôn_nhân và gia_đình",
    )
    tokens = [t["token"] for t in resp["tokens"]]
    assert tokens == ["hôn_nhân", "và", "gia_đình"]
