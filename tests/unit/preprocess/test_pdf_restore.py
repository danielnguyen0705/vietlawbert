# -*- coding: utf-8 -*-
"""Unit-test pdf_restore: định tuyến native-text (tự sinh PDF, không cần tesseract) + biên rỗng."""
import os
import sys

from preprocess.pdf_restore import restore_pdf

_VN = ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM. Độc lập - Tự do - Hạnh phúc. "
       "THÔNG TƯ quy định mức thu, chế độ thu, nộp, quản lý và sử dụng phí thẩm định. "
       "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng của văn bản pháp luật này.")


def _make_text_pdf() -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    # insert_htmlbox giữ đúng dấu tiếng Việt (insert_text + font mặc định làm hỏng dấu);
    # lặp 4 lần để vượt _MIN_NATIVE_CHARS=200.
    page.insert_htmlbox(fitz.Rect(50, 50, 550, 750), (_VN + " ") * 4)
    return doc.tobytes()


def test_native_text_duoc_dung_truc_tiep_khong_ocr():
    r = restore_pdf(_make_text_pdf(), max_pages=5)
    assert r["method"] == "native_text" and r["status"] == "accepted"
    assert r["vi_ratio"] >= 0.05 and r["chars"] >= 200


def test_bytes_rong_tra_empty():
    r = restore_pdf(b"", max_pages=5)
    assert r["status"] == "empty" and r["method"] == "none" and r["text"] == ""
