# -*- coding: utf-8 -*-
"""Unit-test cho legacy_font_restore: hàng rào §E + restore_record (mock API, không key)."""
import os
import sys

from preprocess.legacy_font_restore import restore_guard, restore_record

_GARBLED = "CQNG HOA XA HQI CHU NGHIA VIPT NAM so 287/2024/NQ-HDND Ldm Ding ndm 2024"
_GOOD = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM số 287/2024/NQ-HĐND Lâm Đồng năm 2024"


def test_guard_chap_nhan_phuc_hoi_sach():
    ok, reason = restore_guard(_GARBLED, _GOOD)
    assert ok, reason


def test_guard_chan_bia_phinh_noi_dung():
    hallucinated = _GOOD + " Nghị quyết này nhằm mục đích quan trọng và cần thiết cho toàn tỉnh trong giai đoạn mới."
    ok, reason = restore_guard(_GARBLED, hallucinated)
    assert not ok and reason.startswith("length_ratio")


def test_guard_chan_mat_so_hieu():
    drop = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM số NQ-HĐND Lâm Đồng"
    ok, reason = restore_guard(_GARBLED, drop)
    assert not ok and reason == "digits_lost"


def test_guard_chan_khi_vi_khong_cai_thien():
    ok, reason = restore_guard(_GARBLED, _GARBLED)      # trả lại y hệt → vi không tăng
    assert not ok and reason.startswith("vi_not_improved")


def test_restore_record_mock_dat():
    raw = {"doc_id": "x1", "doc_number": "287/2024/NQ-HĐND", "full_text": _GARBLED}
    r = restore_record(raw, lambda _t: _GOOD)
    assert r["status"] == "restored" and r["needs_human_review"] is True
    assert r["vi_after"] > r["vi_before"]


def test_restore_record_api_loi_giu_goc():
    def boom(_t): raise RuntimeError("down")
    r = restore_record({"doc_id": "x2", "full_text": _GARBLED}, boom)
    assert r["status"] == "unresolved" and r["reason"].startswith("api_error")
    assert r["restored_text"] is None
