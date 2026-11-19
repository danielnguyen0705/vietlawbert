# -*- coding: utf-8 -*-
"""
Unit-test cho profile_sources.profile_records (HÀM THUẦN) + percentiles.
Nguyên tắc dự án: làm đến đâu kiểm thử đến đó. Dữ liệu TỔNG HỢP tự tạo, KHÔNG dùng
nội dung thật/VLQA, không đụng DB/đĩa.
"""
import os
import sys

from quality.corpus_audit.profile_sources import profile_records, percentiles


def test_percentiles_tren_mang_biet_truoc():
    # nearest-rank, 0-based: idx = (p*n)//100 với n=10
    assert percentiles(list(range(1, 11))) == {1: 1, 5: 1, 25: 3, 50: 6, 75: 8, 95: 10}
    assert percentiles([]) == {1: 0, 5: 0, 25: 0, 50: 0, 75: 0, 95: 0}


def _recs():
    return [
        # 3 content_source khác nhau, số hiệu VALID, có text tốt
        {"doc_id": "a1", "doc_number": "100/2019/NĐ-CP", "content_source": "detail_html",
         "html_status": "VALID", "ocr_status": None, "rescue_status": None,
         "doc_type": None, "vi_ratio": 0.30, "length": 2600},
        {"doc_id": "a2", "doc_number": "15/2024/QH15", "content_source": "legacy_print_html",
         "html_status": "VALID", "ocr_status": None, "rescue_status": None,
         "doc_type": None, "vi_ratio": 0.28, "length": 1800},
        {"doc_id": "a3", "doc_number": "72/2026/QĐ-UBND", "content_source": "fallback_html",
         "html_status": "VALID", "ocr_status": None, "rescue_status": None,
         "doc_type": None, "vi_ratio": 0.25, "length": 900},
        # bản HTML rỗng + OCR_PENDING + không có text để đo (vi_ratio None, length 0)
        {"doc_id": "a4", "doc_number": "66/2026/QĐ-UBND", "content_source": None,
         "html_status": "EMPTY", "ocr_status": "OCR_PENDING", "rescue_status": None,
         "doc_type": None, "vi_ratio": None, "length": 0},
        # dưới ngưỡng tiếng Việt (văn bản ngoại/bảng)
        {"doc_id": "a5", "doc_number": "05/2020/TT-BXD", "content_source": "detail_html",
         "html_status": "VALID", "ocr_status": None, "rescue_status": None,
         "doc_type": None, "vi_ratio": 0.01, "length": 300},
        # doc_number CONTAMINATED (rò rỉ mô tả kỳ họp) + THIẾU nhiều trường (test .get)
        {"doc_id": "a6", "doc_number": "Nghị quyết về việc kỳ họp thứ 3 khóa 15"},
    ]


def test_tong_nhom_bang_tong_ban_ghi():
    r = profile_records(_recs())
    n = len(_recs())
    assert r["total"] == n
    for key in ("content_source", "html_status", "ocr_status", "rescue_status",
                "doc_type", "doc_number_class"):
        assert sum(r[key].values()) == n, key


def test_ban_thieu_truong_khong_sap_va_phan_loai_dung():
    r = profile_records(_recs())
    # a6 thiếu content_source/length/vi_ratio... không được sập; rơi vào nhóm (null)
    assert r["content_source"].get("(null)") == 2          # a4 (None) + a6 (thiếu)
    assert r["doc_number_class"].get("CONTAMINATED") == 1  # a6
    assert r["doc_number_class"].get("VALID") == 5
    assert r["no_text_records"] == 2                       # a4 (None) + a6 (thiếu trường)
    assert r["empty_text"] == 2                            # a4 (0) + a6 (thiếu length → 0)
    assert r["below_vi_0_05_overall"] == 1                 # a5
    assert r["unique_doc_number"] == 6


def test_phan_vi_va_bo_theo_nguon_nhat_quan():
    r = profile_records(_recs())
    assert set(r["by_content_source"]) == {"detail_html", "legacy_print_html",
                                           "fallback_html", "(null)"}
    assert sum(d["count"] for d in r["by_content_source"].values()) == r["total"]
    # detail_html có a1 (vi .30) và a5 (vi .01) → có đúng 1 bản dưới ngưỡng
    assert r["by_content_source"]["detail_html"]["below_vi_0_05"] == 1
