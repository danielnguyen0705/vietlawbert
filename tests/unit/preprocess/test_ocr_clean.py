# -*- coding: utf-8 -*-
"""Unit-test ocr_clean: bỏ chữ ký số/rác, nối dòng gãy, giữ tiêu đề + đầu mục đứng riêng."""
import os
import sys

from preprocess.ocr_clean import clean_ocr_text

_RAW = """kau Cơ quan phát hành: VĂN PHÒNG CHÍNH PHỦ
«
CHÍNH PHỦ CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định cơ chế hoạt động, cơ chế quản lý tài chính đặc thù áp
dụng đối với Công ty mẹ - Tập đoàn Điện lực Việt Nam.
sÀ + Lễ ` Ká ` Ã - vá ~ -
1. Cơ quan đại diện chủ sở hữu quyết định các nội dung thuộc thẩm quyền theo
quy định của pháp luật hiện hành.
"""


def test_bo_chu_ky_va_rac_noi_dong_giu_tieu_de():
    clean, st = clean_ocr_text(_RAW)
    lines = clean.splitlines()
    # 1) chữ ký số + dòng trang trí "«" + dòng rác "sÀ..." bị loại
    assert st["dropped_signature"] >= 1 and st["dropped_junk"] >= 2
    assert "Cơ quan phát hành" not in clean and "sÀ" not in clean
    # 2) tiêu đề Điều đứng riêng một dòng
    assert "Điều 1. Phạm vi điều chỉnh" in lines
    # 3) đoạn thân bị cắt 2 dòng được nối thành 1 (không còn dòng kết thúc bằng "áp")
    assert not any(l.rstrip().endswith("áp") for l in lines)
    body = next(l for l in lines if l.startswith("Nghị định này"))
    assert "áp dụng đối với Công ty mẹ" in body
    # 4) đầu mục "1." bắt đầu đoạn mới, nối phần thân theo sau
    khoan = next(l for l in lines if l.startswith("1. Cơ quan đại diện"))
    assert "quy định của pháp luật hiện hành." in khoan


def test_rong_tra_rong():
    clean, st = clean_ocr_text("")
    assert clean == "" and st["lines_out"] == 0
