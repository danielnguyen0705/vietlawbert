"""Unit test cơ chế kiểm định chất lượng tệp cào và tính toàn vẹn ngôn ngữ."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality.crawl_audit import evaluate_linguistic_quality, audit_crawl
from artifacts.canonical import write_jsonl


def test_linguistic_purity_evaluation():
    # Tiếng Việt chuẩn
    valid_vn = "Căn cứ Luật Giao thông đường bộ số 23/2008/QH12 quy định về xử phạt vi phạm."
    res_valid = evaluate_linguistic_quality(valid_vn)
    assert res_valid["is_valid"] is True
    assert res_valid["vietnamese_ratio"] >= 0.05

    # Lỗi vỡ font / ký tự điều khiển
    corrupted = "Văn bản quy phạm \x00\ufffd lỗi font nặng " * 5
    res_corrupted = evaluate_linguistic_quality(corrupted)
    assert res_corrupted["has_encoding_error"] is True
    assert res_corrupted["is_valid"] is False

    # Tiếng nước ngoài không có dấu
    foreign = "This document is purely written in English without any Vietnamese diacritics."
    res_foreign = evaluate_linguistic_quality(foreign)
    assert res_foreign["vietnamese_ratio"] < 0.05
    assert res_foreign["is_valid"] is False


def test_audit_crawl_detects_corrupted_encoding(tmp_path):
    shard_path = tmp_path / "test_shard.jsonl"
    write_jsonl(
        shard_path,
        [
            {"item_id": "1", "html_status": "VALID", "html_raw": "Nội dung chuẩn tiếng Việt có dấu rất đầy đủ và hợp lệ."},
            {"item_id": "2", "html_status": "VALID", "html_raw": "Nội dung \x00 dính rác " * 20},
        ],
    )
    report = audit_crawl(shard_path, expected_documents=2)
    assert report["linguistic_quality_rejected"] == 1
    assert report["records"] == 2