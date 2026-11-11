# -*- coding: utf-8 -*-
"""Kiểm text_cleaner: khối chữ ký (T/M, K/T, chức danh, tên người ký) bị loại; nội dung giữ nguyên."""
from preprocess.text_cleaner import clean_boilerplate


def test_khoi_ky_bi_loai_ca_3491():
    text = ("Điều 7. Tổ chức thực hiện\n"
            "Chánh Văn phòng và các đơn vị liên quan chịu trách nhiệm thi hành quyết định này./.\n"
            "T/M ỦY BAN NHÂN DÂN THÀNH PHỐ\n"
            "K/T CHỦ TỊCH\n"
            "PHÓ CHỦ TỊCH\n"
            "Nguyễn Văn Chí\n")
    out = clean_boilerplate(text)
    assert "T/M ỦY BAN" not in out
    assert "PHÓ CHỦ TỊCH" not in out
    assert "Nguyễn Văn Chí" not in out          # tên người ký cũng bị loại
    assert "Tổ chức thực hiện" in out            # nội dung điều vẫn còn


def test_noi_dung_dieu_thuong_khong_bi_cat():
    text = ("Điều 3. Nguyên tắc báo cáo\n"
            "1. Báo cáo phải trung thực, đầy đủ, kịp thời theo quy định của pháp luật.\n"
            "2. Người đứng đầu đơn vị chịu trách nhiệm về nội dung báo cáo.\n")
    out = clean_boilerplate(text)
    assert "trung thực, đầy đủ, kịp thời" in out
    assert "Người đứng đầu đơn vị" in out
