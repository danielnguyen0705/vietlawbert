# -*- coding: utf-8 -*-
"""Unit-test appendix_extract.dedup_paragraphs: khử trùng đoạn across thành phần, giữ lần đầu."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from preprocess.appendix_extract import dedup_paragraphs  # noqa: E402


def test_khu_trung_doan_across_mau_hsmt():
    # 2 mẫu HSMT: nhiều đoạn boilerplate trùng nhau, vài đoạn riêng
    boiler = "Điều 1. Phạm vi áp dụng của hồ sơ mời thầu theo quy định pháp luật hiện hành."
    m1 = {"member": "01A.docx", "text": boiler + "\nĐoạn riêng của mẫu nội khối một túi hồ sơ."}
    m2 = {"member": "01B.docx", "text": boiler.upper() + "   \nĐoạn riêng của mẫu quốc tế hai túi hồ sơ."}
    unique, st = dedup_paragraphs([m1, m2])
    # boiler (dù khác hoa/thường/space) chỉ còn 1; 2 đoạn riêng giữ lại → 3 duy nhất / 4 tổng
    assert st["total_paragraphs"] == 4
    assert st["unique_paragraphs"] == 3 and st["dup_removed"] == 1
    assert unique[0]["source"] == "01A.docx"      # giữ lần xuất hiện đầu
    assert any("nội khối" in u["text"] for u in unique)
    assert any("quốc tế" in u["text"] for u in unique)


def test_bo_doan_qua_ngan():
    m = {"member": "x", "text": "ok\nĐoạn đủ dài để được giữ lại trong kết quả khử trùng nhé."}
    unique, st = dedup_paragraphs([m], min_len=20)
    assert st["total_paragraphs"] == 1 and st["unique_paragraphs"] == 1


def test_build_appendix_chunks_cha_con(tmp_path):
    import io
    import zipfile
    import docx
    from preprocess.appendix_extract import build_appendix_chunks

    d = docx.Document()
    for i in range(60):
        d.add_paragraph(f"Điều {i}. Đoạn nội dung phụ lục tiếng Việt đủ dài để đóng gói chunk số {i} trong mẫu hồ sơ mời thầu.")
    bio = io.BytesIO(); d.save(bio)
    zpath = tmp_path / "ph.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Final 1/01A.docx", bio.getvalue())

    chunks, stats = build_appendix_chunks(
        str(zpath), "DOC1", {"status": "Còn hiệu lực", "doc_number": "1/2026/TT-BTC"})
    assert chunks, "phải tạo được chunk con"
    assert all(c["doc_id"] == "DOC1" for c in chunks)                       # gắn văn bản cha
    assert all(c["metadata"]["parent_doc_id"] == "DOC1" for c in chunks)    # quan hệ cha–con
    assert all(c["metadata"]["is_appendix"] is True for c in chunks)
    assert all(c["metadata"]["status"] == "Còn hiệu lực" for c in chunks)   # kế thừa hiệu lực
    assert chunks[0]["chunk_id"] == "DOC1_appendix_0"
    assert len(chunks) >= 2                                                  # có đóng gói nhiều chunk
