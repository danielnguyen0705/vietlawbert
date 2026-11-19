# -*- coding: utf-8 -*-
"""Unit-test cho parser_sanity: structural_class + check_doc. Synthetic, không DB/đĩa."""
import os
import sys

from quality.corpus_audit.parser_sanity import structural_class, check_doc, is_boilerplate_line


def test_structural_class_theo_cap_cao_nhat():
    assert structural_class("Phần thứ nhất\nChương I\nĐiều 1.") == "co_phan"
    assert structural_class("Chương I\nMục 1\nĐiều 1.") == "co_muc"
    assert structural_class("Chương I\nĐiều 1.") == "co_chuong"
    assert structural_class("Điều 1. Phạm vi\nĐiều 2. Đối tượng") == "dieu_phang"
    assert structural_class("Quyết định về việc ban hành kế hoạch công tác năm.") == "khong_dieu"


def _chunk(art, body):
    return {"article_num": art, "content": f"[META] x [HIERARCHY] y [CONTENT] {body}"}


def test_check_doc_dat_het_khi_khop():
    txt = "Điều 1. Phạm vi điều chỉnh của văn bản.\nĐiều 2. Đối tượng áp dụng cụ thể."
    chunks = [_chunk("1", "Điều 1. Phạm vi điều chỉnh của văn bản."),
              _chunk("2", "Điều 2. Đối tượng áp dụng cụ thể.")]
    r = check_doc(chunks, txt)
    assert r["article_match"] and r["coverage_ok"] and r["monotonic"] and r["no_empty_chunk"]
    assert r["structural_class"] == "dieu_phang"


def test_check_doc_bat_loi():
    txt = "Điều 1. Nội dung điều một đầy đủ.\nĐiều 2. Nội dung điều hai đầy đủ."
    # thiếu Điều 2 (mất chữ) + thứ tự giảm + 1 chunk rỗng
    chunks = [_chunk("2", "Điều 2. Nội dung điều hai đầy đủ."),
              _chunk("1", "x")]
    r = check_doc(chunks, txt)
    assert r["article_match"] is True          # tập {1,2} == {1,2} (vẫn đủ số Điều)
    assert r["monotonic"] is False             # 2 rồi 1 → giảm
    assert r["no_empty_chunk"] is False        # chunk 'x' < 10 ký tự
    assert r["coverage"] < 0.95                # mất phần lớn chữ Điều 1


def test_khong_dieu_van_dat_khop():
    txt = "Căn cứ Luật Tổ chức chính quyền địa phương ban hành kế hoạch công tác."
    chunks = [{"article_num": "", "content": "[CONTENT] " + txt}]
    r = check_doc(chunks, txt)
    assert r["article_match"] is True          # 0 Điều nguồn == 0 Điều parser
    assert r["structural_class"] == "khong_dieu"


def test_boilerplate_va_tieu_de_hierarchy_khong_tinh_la_mat_chu():
    assert is_boilerplate_line("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    assert is_boilerplate_line("Độc lập - Tự do - Hạnh phúc")
    assert is_boilerplate_line("Hà Nội, ngày 18 tháng 11 năm 2026")
    txt = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "Hà Nội, ngày 18 tháng 11 năm 2026\n"
        "Điều 1. Điều khoản thi hành\n"
        "1. Quyết định này có hiệu lực thi hành kể từ ngày ký ban hành.\n"
        "Nơi nhận:\n"
        "TM. ỦY BAN NHÂN DÂN"
    )
    chunks = [{
        "article_num": "1",
        "hierarchy_path": "Điều 1 (Điều khoản thi hành) > Khoản 1",
        "content": (
            "[META] x [HIERARCHY] Điều 1 (Điều khoản thi hành) > Khoản 1\n"
            "[CONTENT] 1. Quyết định này có hiệu lực thi hành kể từ ngày ký ban hành."
        ),
    }]
    r = check_doc(chunks, txt)
    assert r["coverage"] == 1.0
    assert r["coverage_ok"] is True
    assert r["coverage_raw"] < 0.95            # sàn thô (chưa lọc boilerplate) sẽ bị tụt

