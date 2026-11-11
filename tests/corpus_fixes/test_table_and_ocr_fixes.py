# -*- coding: utf-8 -*-
"""Kiểm tables_to_markdown (giữ cấu trúc bảng) và cổng OCR ocr_broken_ratio."""
from bs4 import BeautifulSoup
from preprocess.html_to_md import tables_to_markdown
from crawler.ocr_runner import ocr_broken_ratio


def test_table_giu_cau_truc_pipe():
    html = ("<div><table>"
            "<tr><th>STT</th><th>Hạng mục</th><th>Cơ cấu (%)</th></tr>"
            "<tr><td>1</td><td>Điều tra thu thập</td><td>6,11</td></tr>"
            "<tr><td>2</td><td>Phân tích đánh giá</td><td>10,99</td></tr>"
            "</table></div>")
    soup = BeautifulSoup(html, "html.parser")
    tables_to_markdown(soup)
    text = soup.get_text("\n", strip=True)
    # số gắn liền hàng (không còn lìa): dòng chứa cả '1' và '6,11'
    assert any("1" in ln and "6,11" in ln for ln in text.splitlines())
    assert "| STT | Hạng mục | Cơ cấu (%) |" in text      # header pipe
    assert "---" in text                                   # separator markdown


def test_ocr_broken_ratio_bat_rac():
    rac = "- O6i vOi t6 chric: T€n t6 chric, <lia chi tru sd chinh, sii Ci6y phdp thAnh lflp"
    assert ocr_broken_ratio(rac) > 0.12


def test_ocr_broken_ratio_van_ban_sach():
    sach = ("Điều 5. Nội dung báo cáo bao gồm tình hình thực hiện công tác phòng chống dịch bệnh "
            "theo quy định của Thông tư 03/2015/TT-BYT và các văn bản liên quan.")
    assert ocr_broken_ratio(sach) < 0.05
